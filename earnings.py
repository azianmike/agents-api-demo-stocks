"""Research recent earnings with the Agents API and download the briefing."""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx2 as httpx
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, OpenAI, OpenAIError

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-6-luna"
FILES = ("earnings.html", "research.json", "earnings-briefing.zip")
log = logging.getLogger(__name__)


class Disconnected(ConnectionError):
    """The stream ended without a confirmed turn outcome."""


def recoverable(error):
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 429) or error.status_code >= 500
    return type(error) is APIError or isinstance(
        error, (APIConnectionError, httpx.TransportError, Disconnected)
    )


def retry(operation, label):
    """Retry reads and downloads; never submit another research request."""
    for attempt in range(7):
        try:
            return operation()
        except (APIError, httpx.TransportError, Disconnected) as error:
            if not recoverable(error) or attempt == 6:
                raise
            delay = min(2 ** (attempt + 1), 30)
            log.warning("%s interrupted (%s); retrying in %ss", label, type(error).__name__, delay)
            time.sleep(delay)


def observe(stream, output, run):
    """Record events until this root turn completes, fails, or loses contact."""
    with (output / "events.jsonl").open("a", encoding="utf-8", buffering=1) as events:
        for event in stream:
            events.write(event.to_json(indent=None) + "\n")
            if event.type == "agent.session.created":
                run["session_id"] = event.session.id
                save_run(output, **run)
                log.info("Session: %s", event.session.id)
            elif event.type.startswith("agent.session.turn.") and hasattr(event, "turn"):
                if event.turn.subagent_id is not None:
                    continue
                if run["turn_id"] is None:
                    run["turn_id"] = event.turn.id
                    save_run(output, **run)
                if event.turn.id != run["turn_id"]:
                    continue
                if event.turn.status == "completed":
                    return run["session_id"], run["turn_id"]
                if event.turn.status in ("failed", "cancelled"):
                    raise RuntimeError(f"Turn {event.turn.status}: {event.turn.error}")
            elif event.type == "agent.session.turn.output_text.done":
                log.info("%s", event.text)
            elif event.type in ("agent.session.failed", "agent.session.environment.failed",
                                "agent.session.requires_action"):
                raise RuntimeError(f"Session needs attention: {event.type}. See {run['session_id']}.")
            elif event.type == "error":
                raise Disconnected("Stream reported an error; check saved state")
    raise Disconnected("Stream ended before the root turn completed")


def resume(client, output, run):
    """Reconnect to saved work, including a completion missed while offline."""
    api = client.beta.agents.sessions

    def reconnect():
        # Subscribe first so completion cannot be lost between checking and listening.
        with api.events.stream(run["session_id"]) as stream:
            session = api.retrieve(run["session_id"])
            if run["turn_id"]:
                turn = api.turns.retrieve(run["turn_id"], session_id=run["session_id"])
            else:
                turn = next((t for t in api.turns.list(run["session_id"], order="asc")
                             if t.subagent_id is None), None)
            if turn:
                run["turn_id"] = turn.id
                save_run(output, **run)
                log.info("Saved turn status: %s", turn.status)
                if turn.status == "completed":
                    return run["session_id"], turn.id
                if turn.status in ("failed", "cancelled"):
                    raise RuntimeError(f"Turn {turn.status}: {turn.error}")
            if session.status in ("failed", "requires_action") or session.required_actions:
                raise RuntimeError(f"Session needs attention: {session.status}. See {session.id}.")
            return observe(stream, output, run)

    return retry(reconnect, "Reconnection")


def run_agent(client, output):
    """Run the prompt, save live events, and return the completed session and turn."""
    now = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")
    prompt = (ROOT / "prompt.txt").read_text().replace("{{AS_OF_DATE}}", now)
    log.info("Researching with %s…", MODEL)
    run = {"session_id": None, "turn_id": None}
    log.info("Dashboard: https://platform.openai.com/logs?api=agents")
    try:
        # An uncertain POST outcome must not silently create a second paid run.
        with client.with_options(max_retries=0).beta.agents.sessions.create(
            agent={
                "model": MODEL,
                "reasoning": {"effort": "high"},
                "instructions": "Research carefully, create the requested files, and verify them.",
                "tools": [{"type": "web_search", "mode": "live"}],
            },
            environment={"type": "openai_hosted", "container_size": "small",
                         "network": {"access": "enabled"}},
            input=prompt,
            stream=True,
        ) as stream:
            return observe(stream, output, run)
    except (APIError, httpx.TransportError, Disconnected) as error:
        if not recoverable(error):
            raise
        if not run["session_id"]:
            raise RuntimeError("Connection lost before a session ID was saved. Check the Agents "
                               "dashboard before starting again; the request may have succeeded.") from error
        log.warning("Connection lost; checking and reconnecting to the existing session.")
        return resume(client, output, run)


def download(client, session_id, turn_id, output):
    """Download the briefing files from this turn only."""
    api = client.beta.agents.sessions.artifacts
    artifacts = retry(lambda: {
        artifact.path: artifact.id
        for artifact in api.list(session_id)
        if artifact.turn_id == turn_id
    }, "Artifact listing")
    missing = []
    for name in FILES:
        artifact_id = artifacts.get(f"/workspace/outputs/{name}")
        if not artifact_id:
            missing.append(name)
            continue
        destination = output / name
        partial = destination.with_suffix(destination.suffix + ".part")

        def fetch():
            with api.with_streaming_response.content(artifact_id, session_id=session_id) as response:
                response.stream_to_file(partial)
            partial.replace(destination)

        retry(fetch, f"Download {name}")
        log.info("Saved %s", destination)
    if missing:
        raise RuntimeError(f"Agent did not publish: {', '.join(missing)}. See session {session_id}.")


def save_run(output, session_id, turn_id=None):
    run = {"session_id": session_id, "turn_id": turn_id}
    temporary = output / "run.json.tmp"
    temporary.write_text(json.dumps(run, indent=2))
    temporary.replace(output / "run.json")


def export_logs(client, session_id, output):
    """Save recorded history and available traces without starting agent work."""
    api = client.beta.agents.sessions
    log.info("Refresh logs: .venv/bin/python earnings.py --logs %s", session_id)
    for name, resource, limit in (
        ("items.jsonl", api.items, 100),
        ("traces.jsonl", api.traces, 1),
    ):
        destination = output / name
        partial = destination.with_suffix(destination.suffix + ".part")

        def fetch():
            count = 0
            with partial.open("w", encoding="utf-8", buffering=1) as file:
                for record in resource.list(session_id, order="asc", limit=limit):
                    file.write(record.to_json(indent=None) + "\n")
                    count += 1
            partial.replace(destination)
            return count

        try:
            count = retry(fetch, f"Export {name}")
            log.info("Saved %s (%d records)", destination, count)
            if name == "traces.jsonl" and not count:
                log.warning("No published traces yet; retry --logs later.")
        except (OpenAIError, httpx.TransportError) as error:
            log.warning("Could not finish exporting %s: %s. Retry --logs later.", name, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", nargs=2, metavar=("SESSION_ID", "TURN_ID"),
                      help="Download an existing completed turn without rerunning research")
    mode.add_argument("--logs", metavar="SESSION_ID",
                      help="Export an existing session's history and traces without rerunning research")
    mode.add_argument("--resume", type=Path, metavar="RUN_JSON",
                      help="Reconnect using saved run.json, then download the existing run's files")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in .env before running.")
    output = (args.resume.resolve().parent if args.resume else
              ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()],
    )
    log.info("Output: %s", output)
    with OpenAI(api_key=key, timeout=60) as client:
        if args.logs:
            save_run(output, args.logs)
            export_logs(client, args.logs, output)
            return
        log.info("Resume after interruption: .venv/bin/python earnings.py --resume %s", output / "run.json")
        try:
            if args.resume:
                run = json.loads(args.resume.read_text())
                if not run.get("session_id"):
                    raise ValueError("run.json has no session_id; check the Agents dashboard.")
                run.setdefault("turn_id", None)
                session_id, turn_id = resume(client, output, run)
            elif args.download:
                session_id, turn_id = args.download
            else:
                session_id, turn_id = run_agent(client, output)
            save_run(output, session_id, turn_id)
            log.info("Download again: .venv/bin/python earnings.py --download %s %s", session_id, turn_id)
            download(client, session_id, turn_id, output)
        finally:
            saved = output / "run.json"
            if saved.exists():
                export_logs(client, json.loads(saved.read_text())["session_id"], output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Run failed")
        raise SystemExit(1) from None
