"""Integration test against real Postgres + pgvector with a fake OpenAI layer."""
import hashlib
import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text

from app.config import Profile, Settings, SourceConfig
from app.db import get_engine, session_scope
from app.enrich import embed_and_dedupe, summarize_pending
from app.llm import RankedItem
from app.main import migrate
from app.models import Article, Base
from app.pipeline import ingest
from app.rank import pick_top_articles
from app.scrapers.base import RawArticle


def fake_vec(text_: str) -> list[float]:
    """Deterministic bag-of-words vector so similar texts land close together."""
    v = [0.0] * 1536
    for w in text_.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 1536] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeLLM:
    usage: dict = {}

    def embed(self, texts):
        return [fake_vec(t) for t in texts]

    def summarize(self, title, content):
        if "BOOM" in title:
            raise RuntimeError("api down")
        return f"Summary of {title}"

    def rerank(self, profile_text, candidates, model=None):
        return [RankedItem(id=c["id"], score=float(len(c["title"]) % 10), reason="r")
                for c in candidates]


@pytest.fixture(autouse=True)
def clean_db():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate()
    yield


def raw(title, url, content="body", hours_ago=1):
    return RawArticle(
        source="S", source_type="rss", url=url, title=title, content=content,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


def test_migration_matches_models():
    with session_scope() as s:
        cols = {r[0] for r in s.execute(text(
            "select column_name from information_schema.columns where table_name='articles'"))}
    assert cols == {c.name for c in Base.metadata.tables["articles"].columns}


def test_full_flow_dedupe_summaries_rank(monkeypatch):
    settings = Settings(top_n=3, openai_api_key="x")
    import app.pipeline as pl

    batch = [
        raw("Acme releases fast new open model", "https://a.com/1", "acme fast open model weights released"),
        raw("Acme releases fast new open model today", "https://b.com/2", "acme fast open model weights released"),
        raw("Totally different vector database post", "https://c.com/3", "pgvector hnsw index tuning guide"),
        raw("BOOM summary fails", "https://d.com/4", "unrelated words zebra quartz"),
    ]
    monkeypatch.setattr(pl, "fetch_all", lambda *a, **k: [(SourceConfig(name="S", url="x"), batch, None)])

    with session_scope() as s:
        stats = ingest(s, settings, [SourceConfig(name="S", url="x")])
    assert stats["S"]["new"] == 4

    # Re-ingesting the same URLs is idempotent
    with session_scope() as s:
        assert ingest(s, settings, [SourceConfig(name="S", url="x")])["S"]["new"] == 0

    llm = FakeLLM()
    with session_scope() as s:
        d = embed_and_dedupe(s, llm, settings, {})
    assert d == {"embedded": 3, "duplicates": 1}

    for _ in range(3):  # BOOM article fails each attempt, then is marked failed
        with session_scope() as s:
            summarize_pending(s, llm, settings)
    with session_scope() as s:
        rows = {a.url: a for a in s.scalars(select(Article))}
        assert rows["https://b.com/2"].status == "duplicate"
        assert rows["https://b.com/2"].duplicate_of == rows["https://a.com/1"].id
        assert rows["https://d.com/4"].status == "failed"
        assert rows["https://a.com/1"].status == "enriched"

    profile = Profile(role="engineer", interests=["vector database", "pgvector"])
    with session_scope() as s:
        picked = pick_top_articles(s, llm, settings, profile)
        assert {a.url for a in picked} == {"https://a.com/1", "https://c.com/3"}
        assert all(a.relevance_score is not None for a in picked)


def test_rate_limited_summaries_do_not_use_up_attempts():
    import httpx
    import openai

    class RateLimitedLLM(FakeLLM):
        def summarize(self, title, content):
            resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
            raise openai.RateLimitError("quota", response=resp, body=None)

    settings = Settings(openai_api_key="x")
    with session_scope() as s:
        s.add(Article(
            source="S", source_type="rss", url="https://r.com/1", url_hash="h1", title="T",
            content="c", published_at=datetime.now(timezone.utc), status="embedded", attempts=0,
            created_at=datetime.now(timezone.utc),
        ))

    for _ in range(5):  # more runs than MAX_ATTEMPTS
        with session_scope() as s:
            stats = summarize_pending(s, RateLimitedLLM(), settings)
        assert stats["rate_limited"] == 1

    with session_scope() as s:
        a = s.scalars(select(Article)).one()
        assert a.status == "embedded" and a.attempts == 0

    with session_scope() as s:  # once the quota recovers it is summarized normally
        assert summarize_pending(s, FakeLLM(), settings)["summarized"] == 1