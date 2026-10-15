import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import get_settings, load_profile, load_sources
from app.db import session_scope
from app.digest import render_html, render_text, send_email
from app.enrich import embed_and_dedupe, summarize_pending
from app.llm import LLM
from app.models import Article, PipelineRun
from app.rank import pick_top_articles
from app.scrapers import fetch_all
from app.scrapers.base import url_hash

log = logging.getLogger(__name__)


def ingest(session, settings, sources) -> dict:
    stats = {}
    for source, articles, error in fetch_all(sources, settings.lookback_hours, settings.worker_count):
        inserted = 0
        if articles:
            rows = [
                {
                    "source": a.source,
                    "source_type": a.source_type,
                    "url": a.url,
                    "url_hash": url_hash(a.url),
                    "title": a.title,
                    "content": a.content,
                    "published_at": a.published_at,
                    "status": "new",
                    "attempts": 0,
                    "created_at": datetime.now(timezone.utc),
                }
                for a in articles
            ]
            result = session.execute(
                pg_insert(Article)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["url_hash"])
                .returning(Article.id)
            )
            inserted = len(result.all())
        stats[source.name] = {"fetched": len(articles), "new": inserted, "error": error}
        if error:
            log.warning("source %s failed: %s", source.name, error)
    return stats


def run(dry_run: bool = False) -> dict:
    settings = get_settings()
    if not settings.api_key:
        key_name = "GEMINI_API_KEY" if settings.provider == "gemini" else "OPENAI_API_KEY"
        raise RuntimeError(f"{key_name} is not set (LLM_PROVIDER={settings.provider})")
    llm = LLM(settings)
    sources, profile = load_sources(), load_profile()
    by_name = {s.name: s for s in sources}

    with session_scope() as session:
        record = PipelineRun()
        session.add(record)
        session.flush()
        run_id = record.id

    stats: dict = {}
    status = "success"
    try:
        with session_scope() as session:
            stats["sources"] = ingest(session, settings, sources)
        with session_scope() as session:
            stats["dedupe"] = embed_and_dedupe(session, llm, settings, by_name)
        with session_scope() as session:
            stats["summaries"] = summarize_pending(session, llm, settings)
        with session_scope() as session:
            picked = pick_top_articles(session, llm, settings, profile)
            stats["digest"] = {"articles": len(picked)}
            if picked and dry_run:
                Path("digest_preview.html").write_text(render_html(picked))
                print(render_text(picked))
            elif picked:
                send_email(settings, picked)
                now = datetime.now(timezone.utc)
                for article in picked:
                    article.sent_at = now
    except Exception:
        status = "failed"
        raise
    finally:
        stats["tokens"] = llm.usage
        with session_scope() as session:
            record = session.get(PipelineRun, run_id)
            record.status, record.stats = status, stats
            record.finished_at = datetime.now(timezone.utc)
        log.info("run %s %s: %s", run_id, status, stats)
    return stats