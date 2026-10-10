import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import httpx2 as httpx
from openai import OpenAI, OpenAIError

from earnings import FILES, download, run_agent


class EarningsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def stream_client(self, ending):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["agent"]["model"], "gpt-6-luna")
            self.assertEqual(payload["agent"]["reasoning"], {"effort": "high"})
            self.assertEqual(payload["agent"]["tools"], [{"type": "web_search", "mode": "live"}])
            self.assertEqual(payload["environment"]["type"], "openai_hosted")
            self.assertNotIn("{{AS_OF_DATE}}", payload["input"])
            self.assertNotIn("OPENAI_API_KEY", request.content.decode())
            messages = [
                {"type": "agent.session.created", "session": {"id": "sess_1"}},
                {"type": "agent.session.turn.created", "session_id": "sess_1", "turn_id": "turn_1",
                 "turn": {"id": "turn_1", "session_id": "sess_1", "subagent_id": None, "status": "in_progress"}},
                *ending,
            ]
            for event in messages:
                if "turn" in event:
                    event.setdefault("turn_id", event["turn"].setdefault("id", "turn_1"))
                    event["turn"].setdefault("session_id", "sess_1")
                    event["turn"].setdefault("status", event["type"].rsplit(".", 1)[-1])
            body = ": keepalive\n\n" + "".join(
                "data: " + json.dumps(event, indent=2).replace("\n", "\ndata: ") + "\n\n"
                for event in messages
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        return OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_waits_for_root_completion(self):
        with self.stream_client([
            {"type": "agent.session.turn.completed", "turn": {"id": "child", "subagent_id": "sub_1"}},
            {"type": "agent.session.idle"},
            {"type": "agent.session.turn.completed", "turn": {"id": "turn_1", "subagent_id": None}},
            {"type": "agent.session.idle"},
        ]) as client:
            self.assertEqual(run_agent(client, self.output), ("sess_1", "turn_1"))

    def test_idle_or_failed_turn_is_not_success(self):
        for ending in ([{"type": "agent.session.idle"}],
                       [{"type": "agent.session.turn.failed", "turn": {"subagent_id": None}}]):
            with self.subTest(ending=ending), self.stream_client(ending) as client:
                with self.assertRaises(OpenAIError):
                    run_agent(client, self.output)
                self.assertEqual(json.loads((self.output / "run.json").read_text()),
                                 {"session_id": "sess_1", "turn_id": "turn_1"})

    def test_download_paginates_and_filters_turn_and_path(self):
        def handler(request):
            if request.url.path.endswith("/content"):
                self.assertNotIn("wrong", request.url.path)
                return httpx.Response(200, content=b"downloaded bytes")
            if "after" not in request.url.params:
                return httpx.Response(200, json={"data": [
                    {"id": "wrong", "turn_id": "old", "path": "/workspace/outputs/earnings.html"},
                    {"id": "wrong-path", "turn_id": "turn_1", "path": "/workspace/outputs/../../.env"},
                ], "has_more": True, "last_id": "wrong-path"})
            self.assertEqual(request.url.params["after"], "wrong-path")
            return httpx.Response(200, json={"data": [
                {"id": name, "turn_id": "turn_1", "path": f"/workspace/outputs/{name}"}
                for name in FILES
            ], "has_more": False})
        with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            download(client, "sess_1", "turn_1", self.output)
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), sorted(FILES))
        self.assertTrue(all((self.output / name).read_bytes() == b"downloaded bytes" for name in FILES))

    def test_missing_artifacts_fail_clearly(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"data": [], "has_more": False}))
        with OpenAI(api_key="test", http_client=httpx.Client(transport=transport)) as client:
            with self.assertRaisesRegex(RuntimeError, "did not publish"):
                download(client, "sess_1", "turn_1", self.output)

    def test_interrupted_download_preserves_existing_file(self):
        class BrokenStream(httpx.SyncByteStream):
            def __iter__(self):
                yield b"partial content"
                raise httpx.ReadError("Connection lost")

        def handler(request):
            if request.url.path.endswith("/content"):
                return httpx.Response(200, stream=BrokenStream())
            return httpx.Response(200, json={"data": [
                {"id": "artifact_1", "turn_id": "turn_1", "path": "/workspace/outputs/earnings.html"}
            ], "has_more": False})

        destination = self.output / "earnings.html"
        destination.write_text("original")
        with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            with self.assertRaises(httpx.ReadError):
                download(client, "sess_1", "turn_1", self.output)
        self.assertEqual(destination.read_text(), "original")
        self.assertTrue((self.output / "earnings.html.part").exists())


if __name__ == "__main__":
    unittest.main()
