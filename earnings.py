"""Research recent earnings with the Agents API and download the briefing."""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-6-luna"
FILES = ("earnings.html", "research.json", "earnings-briefing.zip")
log = logging.getLogger(__name__)


def run_agent(client, output):
    """Run the prompt, save live events, and return the completed session and turn."""
    now = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")
    prompt = (ROOT / "prompt.txt").read_text().replace("{{AS_OF_DATE}}", now)
    log.info("Researching with %s…", MODEL)
    with (
        (output / "events.jsonl").open("w", encoding="utf-8", buffering=1) as events,
        client.beta.agents.sessions.create(
            agent={
                "model": MODEL,
                "reasoning": {"effort": "high"},
                "instructions": "Research carefully, create the requested files, and verify them.",
                "tools": [{"type": "web_search", "mode": "live"}],
            },
            environment={"type": "openai_hosted", "network": {"access": "enabled"}},
            input=prompt,
            stream=True,
        ).with_result_collection() as stream,
    ):
        for event in stream:
            events.write(event.to_json(indent=None) + "\n")
            if event.type == "agent.session.created":
                save_run(output, event.session.id)
                log.info("Session: %s", event.session.id)
                log.info("Dashboard: https://platform.openai.com/logs?api=agents (search by session ID)")
            elif event.type == "agent.session.turn.created" and event.turn.subagent_id is None:
                save_run(output, event.session_id, event.turn_id)
            elif event.type == "agent.session.turn.output_text.done":
                log.info("%s", event.text)
            elif event.type in ("agent.session.failed", "agent.session.requires_action"):
                break
        result = stream.get_final_result()
    return result.session_id, result.turn_id


def download(client, session_id, turn_id, output):
    """Download the briefing files from this turn only."""
    api = client.beta.agents.sessions.artifacts
    artifacts = {
        artifact.path: artifact.id
        for artifact in api.list(session_id)
        if artifact.turn_id == turn_id
    }
    missing = []
    for name in FILES:
        artifact_id = artifacts.get(f"/workspace/outputs/{name}")
        if not artifact_id:
            missing.append(name)
            continue
        destination = output / name
        partial = destination.with_suffix(destination.suffix + ".part")
        with api.with_streaming_response.content(artifact_id, session_id=session_id) as response:
            response.stream_to_file(partial)
        partial.replace(destination)
        log.info("Saved %s", destination)
    if missing:
        raise RuntimeError(f"Agent did not publish: {', '.join(missing)}. See session {session_id}.")


def save_run(output, session_id, turn_id=None):
    run = {"session_id": session_id, "turn_id": turn_id}
    (output / "run.json").write_text(json.dumps(run, indent=2))


def export_logs(client, session_id, output):
    """Save recorded history and available traces without starting agent work."""
    api = client.beta.agents.sessions
    log.info("Refresh logs: .venv/bin/python earnings.py --logs %s", session_id)
    for name, resource, limit in (
        ("items.jsonl", api.items, 100),
        ("traces.jsonl", api.traces, 1),
    ):
        destination = output / name
        try:
            count = 0
            with destination.open("w", encoding="utf-8", buffering=1) as file:
                for record in resource.list(session_id, order="asc", limit=limit):
                    file.write(record.to_json(indent=None) + "\n")
                    count += 1
            log.info("Saved %s (%d records)", destination, count)
            if name == "traces.jsonl" and not count:
                log.warning("No published traces yet; retry --logs later.")
        except OpenAIError as error:
            log.warning("Could not finish exporting %s: %s. Retry --logs later.", name, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", nargs=2, metavar=("SESSION_ID", "TURN_ID"),
                      help="Download an existing completed turn without rerunning research")
    mode.add_argument("--logs", metavar="SESSION_ID",
                      help="Export an existing session's history and traces without rerunning research")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in .env before running.")
    output = ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()],
    )
    log.info("Output: %s", output)
    with OpenAI(api_key=key, timeout=600) as client:
        if args.logs:
            save_run(output, args.logs)
            export_logs(client, args.logs, output)
            return
        if args.download:
            session_id, turn_id = args.download
        else:
            session_id, turn_id = run_agent(client, output)
        save_run(output, session_id, turn_id)
        log.info("Download again: .venv/bin/python earnings.py --download %s %s", session_id, turn_id)
        try:
            download(client, session_id, turn_id, output)
        finally:
            export_logs(client, session_id, output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Run failed")
        raise SystemExit(1) from None
