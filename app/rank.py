"""Stage 4: embedding pre-filter against the profile, then LLM re-rank of the shortlist."""
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Profile, Settings
from app.llm import LLM
from app.models import Article

log = logging.getLogger(__name__)


def rerank_with_fallback(
    llm: LLM, settings: Settings, profile_text: str, items: list[dict]
) -> list:
    """Re-rank with the main model; if it keeps failing (for example Gemini 503 overload),
    try the lighter summary model before giving up."""
    models = [settings.rerank_model_name]
    if settings.summary_model_name not in models:
        models.append(settings.summary_model_name)
    for model in models:
        try:
            return llm.rerank(profile_text, items, model=model)
        except Exception as exc:
            log.warning("rerank with %s failed: %s", model, exc)
    return []


def pick_top_articles(
    session: Session, llm: LLM, settings: Settings, profile: Profile
) -> list[Article]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.lookback_hours)
    profile_vec = llm.embed([profile.as_text()])[0]

    candidates = session.scalars(
        select(Article)
        .where(
            Article.status == "enriched",
            Article.embedding_model == settings.embedding_model_name,
            Article.sent_at.is_(None),
            Article.published_at >= cutoff,
        )
        .order_by(Article.embedding.cosine_distance(profile_vec))
        .limit(settings.candidate_n)
    ).all()
    if not candidates:
        return []

    by_id = {a.id: a for a in candidates}
    ranked = rerank_with_fallback(
        llm,
        settings,
        profile.as_text(),
        [
            {"id": a.id, "title": a.title, "source": a.source, "summary": a.summary}
            for a in candidates
        ],
    )
    if not ranked:
        log.warning("rerank unavailable, using embedding order")

    picked: list[Article] = []
    for item in sorted(ranked, key=lambda r: r.score, reverse=True):
        article = by_id.get(item.id)
        if article and article not in picked:
            article.relevance_score = item.score
            picked.append(article)
    if not picked:
        picked = list(candidates)
    return picked[: settings.top_n]