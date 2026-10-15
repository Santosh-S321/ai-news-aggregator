import html
import smtplib
from datetime import date
from email.message import EmailMessage

from app.config import Settings
from app.models import Article


def render_text(articles: list[Article]) -> str:
    lines = [f"AI News Digest - {date.today():%d %b %Y}", ""]
    for i, a in enumerate(articles, 1):
        lines += [f"{i}. {a.title}", f"   {a.source}", f"   {a.summary}", f"   {a.url}", ""]
    return "\n".join(lines)


def render_html(articles: list[Article]) -> str:
    esc = html.escape
    items = "".join(
        f"""<div style="margin:0 0 22px">
  <div style="font-size:12px;color:#6b7280">{esc(a.source)}</div>
  <a href="{esc(a.url, quote=True)}" style="font-size:17px;font-weight:600;color:#111827;text-decoration:none">{i}. {esc(a.title)}</a>
  <p style="margin:6px 0 0;color:#374151;line-height:1.5">{esc(a.summary or "")}</p>
</div>"""
        for i, a in enumerate(articles, 1)
    )
    return f"""<div style="font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:640px;margin:auto">
<h2 style="margin-bottom:4px">AI News Digest</h2>
<div style="color:#6b7280;margin-bottom:24px">{date.today():%d %b %Y}</div>
{items}</div>"""


def send_email(settings: Settings, articles: list[Article]) -> None:
    if not (settings.smtp_user and settings.smtp_password and settings.email_to):
        raise RuntimeError("Set SMTP_USER, SMTP_PASSWORD and EMAIL_TO to send the digest")
    msg = EmailMessage()
    msg["Subject"] = f"AI News Digest - {date.today():%d %b %Y}"
    msg["From"] = settings.email_from or settings.smtp_user
    msg["To"] = settings.email_to
    msg.set_content(render_text(articles))
    msg.add_alternative(render_html(articles), subtype="html")
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(msg)
