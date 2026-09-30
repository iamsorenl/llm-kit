import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_kit import (FakeChat, DailyCap, LlmError, assert_free_model, compat_chat,
                     ollama_chat, parse_json_reply)

STATE = {"handler": None, "seen": []}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["seen"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        status, out, hang = STATE["handler"](body)
        if hang:
            threading.Event().wait(1)
            return
        data = out if isinstance(out, str) else json.dumps(out)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data.encode())


class KitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    msgs = [{"role": "user", "content": "hi"}]

    def kind(self, fn):
        with self.assertRaises(LlmError) as cm:
            fn()
        return cm.exception.kind

    def test_ollama_ok_and_optional_fields(self):
        STATE["handler"] = lambda b: (200, {"message": {"content": "x"}, "prompt_eval_count": 4, "eval_count": 2}, False)
        r = ollama_chat(self.msgs, "m", url=self.base, format={"type": "object"}, think=False, num_ctx=8192)
        self.assertEqual((r.content, r.in_tokens, r.out_tokens), ("x", 4, 2))
        sent = STATE["seen"][-1]
        self.assertEqual(sent["path"], "/api/chat")
        self.assertEqual(sent["body"]["think"], False)
        self.assertEqual(sent["body"]["options"]["num_ctx"], 8192)
        ollama_chat(self.msgs, "m", url=self.base)
        self.assertNotIn("think", STATE["seen"][-1]["body"])
        self.assertNotIn("format", STATE["seen"][-1]["body"])

    def test_ollama_errors(self):
        STATE["handler"] = lambda b: (500, {"error": "boom"}, False)
        self.assertEqual(self.kind(lambda: ollama_chat(self.msgs, "m", url=self.base)), "api")
        STATE["handler"] = lambda b: (200, "nope", False)
        self.assertEqual(self.kind(lambda: ollama_chat(self.msgs, "m", url=self.base)), "bad-response")
        STATE["handler"] = lambda b: (200, {}, True)
        self.assertEqual(self.kind(lambda: ollama_chat(self.msgs, "m", url=self.base, timeout=0.2)), "timeout")
        self.assertEqual(self.kind(lambda: ollama_chat(self.msgs, "m", url="http://127.0.0.1:1")), "unreachable")

    def test_compat_json_downgrade_and_auth(self):
        calls = []

        def h(b):
            calls.append(b)
            if "response_format" in b:
                return 400, {"error": {"message": "no json mode"}}, False
            return 200, {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5}}, False

        STATE["handler"] = h
        r = compat_chat(self.msgs, base_url=self.base, model="m", api_key="k", json_mode=True)
        self.assertEqual((r.content, r.in_tokens, len(calls)), ("ok", 5, 2))
        self.assertEqual(STATE["seen"][-1]["auth"], "Bearer k")
        self.assertEqual(STATE["seen"][-1]["path"], "/chat/completions")

    def test_compat_downgrade_422_not_429(self):
        calls = []

        def h(b):
            calls.append(b)
            return (422, {}, False) if "response_format" in b else (200, {"choices": [{"message": {"content": "ok"}}]}, False)

        STATE["handler"] = h
        self.assertEqual(compat_chat(self.msgs, base_url=self.base, model="m", json_mode=True).content, "ok")
        self.assertEqual(len(calls), 2)
        calls.clear()
        STATE["handler"] = lambda b: (calls.append(b), (429, {}, False))[1]
        self.assertEqual(self.kind(lambda: compat_chat(self.msgs, base_url=self.base, model="m", json_mode=True)), "rate-limit")
        self.assertEqual(len(calls), 1)

    def test_compat_errors(self):
        self.assertEqual(self.kind(lambda: compat_chat(self.msgs, base_url=self.base, model="m", api_key="")), "no-key")
        for status, kind in [(401, "auth"), (429, "rate-limit"), (503, "api")]:
            STATE["handler"] = lambda b, s=status: (s, {}, False)
            self.assertEqual(self.kind(lambda: compat_chat(self.msgs, base_url=self.base, model="m")), kind)
        STATE["handler"] = lambda b: (200, {"choices": []}, False)
        self.assertEqual(self.kind(lambda: compat_chat(self.msgs, base_url=self.base, model="m")), "bad-response")
        STATE["handler"] = lambda b: (200, {}, True)
        self.assertEqual(self.kind(lambda: compat_chat(self.msgs, base_url=self.base, model="m", timeout=0.2)), "timeout")

    def test_parse_json_reply(self):
        self.assertEqual(parse_json_reply('```json\n{"a": 1}\n```'), ({"a": 1}, None))
        self.assertIsNone(parse_json_reply("nope")[0])
        self.assertEqual(parse_json_reply('{"a": 1}', lambda v: None if "b" in v else "missing b"), (None, "missing b"))

    def test_guard_cap_fake(self):
        for m in ["gpt-oss-20b", "llama-3.3-70b-versatile", "gemma4:12b"]:
            assert_free_model(m)
        for m in ["gpt-4o", "claude-opus-5-5", "o3-mini"]:
            self.assertRaises(ValueError, assert_free_model, m)
        cap = DailyCap(2)
        self.assertEqual([cap.try_consume() for _ in range(3)], [True, True, False])
        fake = FakeChat(["a"])
        self.assertEqual(fake(self.msgs).content, "a")
        self.assertRaises(RuntimeError, fake, self.msgs)


if __name__ == "__main__":
    unittest.main()
