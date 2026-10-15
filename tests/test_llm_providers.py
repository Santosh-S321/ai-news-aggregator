"""Exercise the real OpenAI SDK against a local fake OpenAI-compatible server,
for both provider settings. Verifies request shapes and response handling."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.config import GEMINI_BASE_URL, Settings
from app.llm import LLM, fit_embedding

SEEN: list[dict] = []
MODE = {"null_index": False, "reject_schema": False}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if self.path.endswith("/embeddings"):
            n = 3072 if "gemini" in body["model"] else 1536
            data = [{"object": "embedding", "index": None if MODE["null_index"] else i,
                     "embedding": [float(i + 1)] * n} for i in range(len(body["input"]))]
            out = {"object": "list", "data": data, "model": body["model"],
                   "usage": {"prompt_tokens": 7, "total_tokens": 7}}
        else:
            rf = body.get("response_format", {})
            if rf.get("type") == "json_schema":
                if MODE["reject_schema"]:
                    err = json.dumps({"error": {"message": "unsupported schema", "type": "invalid_request_error"}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                wants_items = "items" in rf["json_schema"]["schema"]["properties"]
            else:
                wants_items = '"items"' in body["messages"][0]["content"]
            if wants_items:
                content = {"items": [{"id": 1, "score": 15, "reason": "r"}]}
            else:
                content = {"summary": "  A short summary.  "}
            if rf.get("type") == "json_object":
                content = "```json\n" + json.dumps(content) + "\n```"  # fenced, like real models do
            else:
                content = json.dumps(content)
            out = {
                "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            }
        payload = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def make_llm(provider, base):
    s = Settings(llm_provider=provider, openai_api_key="oa-key", gemini_api_key="gm-key", requests_per_minute=0)
    llm = LLM(s)
    from openai import OpenAI
    llm.client = OpenAI(api_key=s.api_key, base_url=base, max_retries=0)  # point at fake server
    return s, llm


@pytest.mark.parametrize("provider,emb_model", [
    ("openai", "text-embedding-3-small"), ("gemini", "gemini-embedding-001")])
def test_provider_calls(server, provider, emb_model):
    SEEN.clear()
    s, llm = make_llm(provider, server)
    assert s.embedding_model_name == emb_model

    vecs = llm.embed(["a", "b"])
    assert len(vecs) == 2 and all(len(v) == 1536 for v in vecs)
    if provider == "gemini":  # truncated from 3072 and re-normalized
        assert abs(sum(x * x for x in vecs[0]) - 1.0) < 1e-6

    assert llm.summarize("T", "body") == "A short summary."
    ranked = llm.rerank("profile", [{"id": 1, "title": "t", "source": "s", "summary": "x"}])
    assert ranked[0].score == 10.0  # out-of-range score is clamped

    models = {r["body"]["model"] for r in SEEN}
    assert s.summary_model_name in models and s.rerank_model_name in models
    assert all(r["auth"] == f"Bearer {s.api_key}" for r in SEEN)
    assert llm.usage[s.summary_model_name]["input"] == 11


def test_provider_config():
    g = Settings(llm_provider="gemini", gemini_api_key="k")
    assert g.base_url == GEMINI_BASE_URL and g.worker_count == 2
    o = Settings(llm_provider="openai", workers=3)
    assert o.base_url is None and o.worker_count == 3
    with pytest.raises(ValueError):
        Settings(llm_provider="nope").provider


def test_fit_embedding_rejects_short_vectors():
    with pytest.raises(ValueError):
        fit_embedding([0.1] * 10)


@pytest.fixture
def mode():
    MODE.update(null_index=False, reject_schema=False)
    yield MODE
    MODE.update(null_index=False, reject_schema=False)


def test_embeddings_with_null_index_keep_response_order(server, mode):
    """Regression: Gemini returned index=None and sorted() crashed."""
    mode["null_index"] = True
    _, llm = make_llm("openai", server)
    vecs = llm.embed(["a", "b", "c"])
    assert [v[0] for v in vecs] == [1.0, 2.0, 3.0]


def test_falls_back_to_json_mode_when_schema_rejected(server, mode):
    mode["reject_schema"] = True
    SEEN.clear()
    _, llm = make_llm("gemini", server)
    assert llm.summarize("T", "body") == "A short summary."
    ranked = llm.rerank("profile", [{"id": 1, "title": "t", "source": "s", "summary": "x"}])
    assert ranked[0].id == 1 and ranked[0].score == 10.0
    kinds = [r["body"]["response_format"]["type"] for r in SEEN]
    assert kinds == ["json_schema", "json_object", "json_schema", "json_object"]


def test_throttle_spaces_calls_to_requests_per_minute():
    import time

    s = Settings(llm_provider="gemini", gemini_api_key="k", requests_per_minute=600)  # 0.1s apart
    llm = LLM(s)
    start = time.monotonic()
    for _ in range(4):
        llm._throttle()
    assert time.monotonic() - start >= 0.29


def test_rpm_defaults_per_provider():
    assert Settings(llm_provider="gemini").rpm_limit == 12
    assert Settings(llm_provider="openai").rpm_limit == 0
    assert Settings(llm_provider="gemini", requests_per_minute=0).rpm_limit == 0
    assert Settings(llm_provider="openai", requests_per_minute=30).rpm_limit == 30