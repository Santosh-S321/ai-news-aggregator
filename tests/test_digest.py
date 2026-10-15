from app.digest import render_html
from app.models import Article


def test_html_digest_escapes_content():
    a = Article(
        source="S", title="<script>alert(1)</script>", url='https://x.com/"q',
        summary="a < b",
    )
    out = render_html([a])
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert 'href="https://x.com/&quot;q"' in out
