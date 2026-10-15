import hashlib
import html
import re
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

USER_AGENT = "Mozilla/5.0 (compatible; ai-news-aggregator/0.1)"
_TRACKING = re.compile(r"^(utm_|fbclid|gclid|ref$|ref_src$|source$)", re.I)


class RawArticle(BaseModel):
    source: str
    source_type: str
    url: str
    title: str
    content: str | None = None
    published_at: datetime


def normalize_url(url: str) -> str:
    """Canonical form so the same story from two feeds collides on url_hash."""
    parts = urlsplit(url.strip())
    query = sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def is_transient(exc: BaseException) -> bool:
    """Retry timeouts, connection errors, 429 and 5xx. A 403/404 will not fix itself."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=8),
    reraise=True,
)
def http_get(url: str, timeout: float = 20) -> httpx.Response:
    resp = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True
    )
    resp.raise_for_status()
    return resp


def parse_feed(data: bytes) -> feedparser.FeedParserDict:
    return feedparser.parse(data)