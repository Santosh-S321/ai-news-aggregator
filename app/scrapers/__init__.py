from concurrent.futures import ThreadPoolExecutor

from app.config import SourceConfig
from app.scrapers.base import RawArticle
from app.scrapers.rss import fetch_rss
from app.scrapers.youtube import fetch_youtube

FETCHERS = {"rss": fetch_rss, "youtube": fetch_youtube}


def _fetch_one(source: SourceConfig, hours: int):
    try:
        return source, FETCHERS[source.type](source, hours), None
    except Exception as exc:  # one broken feed must not stop the run
        return source, [], f"{type(exc).__name__}: {exc}"


def fetch_all(
    sources: list[SourceConfig], hours: int, workers: int = 4
) -> list[tuple[SourceConfig, list[RawArticle], str | None]]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda s: _fetch_one(s, hours), sources))
