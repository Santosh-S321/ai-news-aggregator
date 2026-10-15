from datetime import datetime, timedelta, timezone

from app.config import SourceConfig
from app.scrapers.base import RawArticle, http_get, parse_feed, strip_html


def entry_time(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime(*parsed[:6], tzinfo=timezone.utc) if parsed else None


def entries_to_articles(
    feed, source: SourceConfig, source_type: str, hours: int
) -> list[RawArticle]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    for entry in feed.entries:
        link, title = entry.get("link"), (entry.get("title") or "").strip()
        published = entry_time(entry)
        if not link or not title or published is None or published < cutoff:
            continue
        if "/shorts/" in link:
            continue
        body = entry.content[0].value if entry.get("content") else entry.get("summary", "")
        out.append(
            RawArticle(
                source=source.name,
                source_type=source_type,
                url=link,
                title=title,
                content=strip_html(body) or None,
                published_at=published,
            )
        )
    return out


def fetch_rss(source: SourceConfig, hours: int) -> list[RawArticle]:
    feed = parse_feed(http_get(source.url).content)
    if feed.bozo and not feed.entries:
        raise ValueError(f"unparseable feed: {feed.bozo_exception}")
    return entries_to_articles(feed, source, "rss", hours)
