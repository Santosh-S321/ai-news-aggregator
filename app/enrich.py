"""Stages 2-3: full text, embedding + duplicate detection, summarization."""
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import openai
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, SourceConfig
from app.llm import LLM
from app.models import Article
from app.scrapers.base import http_get
from app.scrapers.youtube import get_transcript

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 3
SHORT_CONTENT = 1500  # below this, try to fetch the full article page


def extract_full_text(url: str) -> str | None:
    import trafilatura

    try:
        return trafilatura.extract(http_get(url).text, include_comments=False)
    except Exception:
        return None


def _gather_text(article: Article, source: SourceConfig | None, settings: Settings) -> str | None:
    """Runs in a worker thread; network only, no DB access."""
    content = article.content or ""
    if article.source_type == "youtube" and settings.enable_youtube_transcripts:
        return get_transcript(article.url) or content
    if source and source.fetch_full_text and len(content) < SHORT_CONTENT:
        return extract_full_text(article.url) or content
    return content


def embed_and_dedupe(
    session: Session, llm: LLM, settings: Settings, sources: dict[str, SourceConfig]
) -> dict:
    new = session.scalars(
        select(Article).where(Article.status == "new").order_by(Article.published_at)
    ).all()
    stats = {"embedded": 0, "duplicates": 0}
    if not new:
        return stats

    with ThreadPoolExecutor(max_workers=settings.worker_count) as pool:
        texts = list(pool.map(lambda a: _gather_text(a, sources.get(a.source), settings), new))
    for article, text in zip(new, texts):
        article.content = text or article.content

    vectors = llm.embed([f"{a.title}\n{(a.content or '')[:1500]}" for a in new])
    window = datetime.now(timezone.utc) - timedelta(days=7)
    max_distance = 1 - settings.dup_similarity

    model_name = settings.embedding_model_name
    for article, vec in zip(new, vectors):
        article.embedding = vec
        article.embedding_model = model_name
        session.flush()
        nearest = session.execute(
            select(Article.id, Article.embedding.cosine_distance(vec).label("dist"))
            .where(
                Article.id != article.id,
                Article.embedding_model == model_name,
                Article.status.in_(["embedded", "enriched"]),
                Article.published_at >= window,
            )
            .order_by("dist")
            .limit(1)
        ).first()
        if nearest and nearest.dist <= max_distance:
            article.status, article.duplicate_of = "duplicate", nearest.id
            stats["duplicates"] += 1
        else:
            article.status = "embedded"
            stats["embedded"] += 1
        session.flush()
    return stats


def summarize_pending(session: Session, llm: LLM, settings: Settings) -> dict:
    pending = session.scalars(
        select(Article).where(Article.status == "embedded", Article.attempts < MAX_ATTEMPTS)
    ).all()
    stats = {"summarized": 0, "failed": 0}

    def work(article: Article):
        try:
            return article.id, llm.summarize(article.title, article.content or article.title), None
        except openai.RateLimitError as exc:
            return article.id, None, ("rate_limit", f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            return article.id, None, ("error", f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=settings.worker_count) as pool:
        results = list(pool.map(work, pending))

    by_id = {a.id: a for a in pending}
    for article_id, summary, error in results:
        article = by_id[article_id]
        if summary:
            article.attempts += 1
            article.summary, article.status = summary, "enriched"
            stats["summarized"] += 1
            continue
        kind, message = error
        stats["failed"] += 1
        if kind == "rate_limit":
            # Quota problem, not a problem with this article: retry on the next run
            # without using up one of its attempts.
            stats["rate_limited"] = stats.get("rate_limited", 0) + 1
            log.warning("rate limited, will retry next run: %s", article.url)
            continue
        article.attempts += 1
        log.warning("summary failed for %s: %s", article.url, message)
        if article.attempts >= MAX_ATTEMPTS:
            article.status = "failed"
    return stats