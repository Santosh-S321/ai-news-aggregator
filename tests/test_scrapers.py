from datetime import datetime, timezone

from app.config import SourceConfig
from app.scrapers.base import normalize_url, parse_feed, strip_html, url_hash
from app.scrapers.rss import entries_to_articles

NOW = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
FEED = f"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>New model released</title><link>https://example.com/post?utm_source=x</link>
<pubDate>{NOW}</pubDate><description>&lt;p&gt;Big &amp;amp; fast&lt;/p&gt;</description></item>
<item><title>Old post</title><link>https://example.com/old</link>
<pubDate>Mon, 01 Jan 2024 00:00:00 +0000</pubDate></item>
<item><title>A short</title><link>https://youtube.com/shorts/abc</link><pubDate>{NOW}</pubDate></item>
</channel></rss>""".encode()


def test_normalize_url_strips_tracking_and_trailing_slash():
    a = normalize_url("HTTPS://Example.com/post/?utm_source=x&b=2&a=1#frag")
    assert a == "https://example.com/post?a=1&b=2"
    assert url_hash("https://example.com/post/") == url_hash("https://example.com/post?ref=tw")


def test_strip_html():
    assert strip_html("<p>Hello &amp; <b>world</b></p>") == "Hello & world"


def test_entries_filtered_by_age_and_shorts():
    feed = parse_feed(FEED)
    items = entries_to_articles(feed, SourceConfig(name="T", url="x"), "rss", hours=48)
    assert [i.title for i in items] == ["New model released"]
    assert items[0].content == "Big & fast"


def test_only_transient_http_errors_are_retried():
    import httpx
    from app.scrapers.base import is_transient

    def status(code):
        req = httpx.Request("GET", "http://x")
        return httpx.HTTPStatusError("e", request=req, response=httpx.Response(code, request=req))

    assert not is_transient(status(403)) and not is_transient(status(404))
    assert is_transient(status(429)) and is_transient(status(502))
    assert is_transient(httpx.ConnectTimeout("t"))