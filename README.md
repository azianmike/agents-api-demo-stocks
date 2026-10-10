# Earnings briefing

A small Python script using OpenAI's Agents API with `gpt-6-luna` and high reasoning effort, live web search,
and a hosted sandbox. It discovers three recent earnings announcements among
leading technology companies, reads reports and investor analysis, and creates
an offline HTML briefing. Edit `prompt.txt` to change the research or design.

## Run

Requires Python 3.10+. Python 3.12 is used in this project.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Set your existing `.env` file to include:

```dotenv
OPENAI_API_KEY=your-api-key
```

```sh
.venv/bin/python earnings.py
```

Progress appears in the terminal. Each run creates a new folder under `outputs/`
containing `earnings.html`, `research.json`, `earnings-briefing.zip`, and `run.json`.
Open the HTML directly, or extract the ZIP to access the source reports too.
The key stays local; only the research prompt is sent to the hosted environment.

To retry a download without paying for another research run, use the IDs saved
in `run.json`:

```sh
.venv/bin/python earnings.py --download SESSION_ID TURN_ID
```

The script uses the official OpenAI Python SDK's `client.beta.agents` interface.
The SDK handles stream parsing, authentication, and artifact pagination.
Python 3.9 installs an older SDK without this interface; use Python 3.10 or newer
when creating the virtual environment. Your project needs Agents
API access and a key with `api.agents.read`, `api.agents.write`, and
`api.responses.write` permissions. A run uses billable model and tool resources.
If the stream disconnects, inspect the saved session before starting another run;
disconnecting the client does not necessarily cancel remote work.

## Logs and traces

Each run saves logs alongside its outputs:

- `run.log`: timestamped progress, download messages, and errors, also shown in the terminal.
- `events.jsonl`: live API events, saved as they arrive, including received events before a failure or disconnect.
- `items.jsonl`: recorded root-agent messages and tool activity, exported after the download attempt.
- `traces.jsonl`: available session trace records, with OTLP data in each record's `otlp` field.

Tracing is enabled by default. Open [Logs → Agents](https://platform.openai.com/logs?api=agents)
and search for the session ID in `run.json` to inspect the timeline and token usage.
Traces and usage can arrive after a turn finishes. Trace export requires organization
enablement and `api.traces.read` or `api.agents.read` permission; an unavailable export
does not prevent briefing downloads.

To retrieve logs later, including after a failed run:

```sh
.venv/bin/python earnings.py --logs SESSION_ID
```

This reads existing history and traces into a new output folder; it does not start
an agent run. Exports include currently published records and follow all pages.
`items.jsonl` covers the root agent; subagents have separate histories. Logs and
traces can include prompts and tool contents and stay under the ignored `outputs/` folder.

References: [Agents API quickstart](https://developers.openai.com/api/docs/guides/agents-api/quickstart),
[files and artifacts](https://developers.openai.com/api/docs/guides/agents-api/environments/files),
[stream events](https://developers.openai.com/api/docs/guides/agents-api/sessions/events),
[observability](https://developers.openai.com/api/docs/guides/agents-api/observability),
[tracing](https://developers.openai.com/api/docs/guides/agents-api/tracing).

## Local checks

```sh
.venv/bin/python -m unittest discover -s tests
```

Tests use mocked HTTP responses and do not call OpenAI or spend API credits.
