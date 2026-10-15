from app.config import SourceConfig
from app.scrapers.base import RawArticle, http_get, parse_feed
from app.scrapers.rss import entries_to_articles


def fetch_youtube(source: SourceConfig, hours: int) -> list[RawArticle]:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={source.channel_id}"
    feed = parse_feed(http_get(url).content)
    return entries_to_articles(feed, source, "youtube", hours)


def get_transcript(url: str) -> str | None:
    """Optional. YouTube often blocks cloud IPs, so failures are expected."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        video_id = url.split("v=")[1].split("&")[0]
        snippets = YouTubeTranscriptApi().fetch(video_id).snippets
        return " ".join(s.text for s in snippets)
    except Exception:
        return None
