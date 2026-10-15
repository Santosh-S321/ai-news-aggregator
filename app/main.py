import argparse
import json
import logging
import sys

from alembic import command
from alembic.config import Config

from app.config import ROOT, get_settings, load_sources


def migrate() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(cfg, "head")


def check_sources() -> int:
    from app.scrapers import fetch_all

    failures = 0
    for source, articles, error in fetch_all(load_sources(), hours=24 * 30):
        if error:
            failures += 1
            print(f"FAIL  {source.name}: {error}")
        else:
            print(f"ok    {source.name}: {len(articles)} items in the last 30 days")
    return 1 if failures else 0


def check_llm() -> int:
    """Tiny end-to-end call to each model so key/provider problems show up fast."""
    from app.llm import LLM

    s = get_settings()
    if not s.api_key:
        print(f"FAIL  no API key set for provider {s.provider!r}")
        return 1
    llm = LLM(s)
    print(f"provider: {s.provider}")
    steps = [
        (f"embeddings ({s.embedding_model_name})",
         lambda: f"{len(llm.embed(['hello world'])[0])} dims"),
        (f"summary ({s.summary_model_name})",
         lambda: llm.summarize("Test post", "A lab released a new open model with a longer context window.")[:80]),
        (f"rerank ({s.rerank_model_name})",
         lambda: str(llm.rerank("Role: engineer", [
             {"id": 1, "title": "t", "source": "s", "summary": "x"}])[0].score)),
    ]
    failures = 0
    for name, fn in steps:
        try:
            print(f"ok    {name}: {fn()}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="ai-news")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="scrape, enrich, rank and email the digest")
    run_p.add_argument("--dry-run", action="store_true", help="print the digest instead of emailing")
    sub.add_parser("migrate", help="apply database migrations")
    sub.add_parser("check-sources", help="test every configured source")
    sub.add_parser("check-llm", help="test the LLM provider, key and models")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    get_settings()
    if args.cmd == "migrate":
        migrate()
    elif args.cmd == "check-sources":
        return check_sources()
    elif args.cmd == "check-llm":
        return check_llm()
    elif args.cmd == "run":
        from app.pipeline import run

        print(json.dumps(run(dry_run=args.dry_run), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())