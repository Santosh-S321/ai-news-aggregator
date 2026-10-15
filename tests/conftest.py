"""Test setup. The database tests DROP the whole public schema, so they must never run
against a real database. Tests always use TEST_DATABASE_URL (never DATABASE_URL), and
refuse to start unless the database name contains "test"."""
import os
from pathlib import Path

from dotenv import dotenv_values

# TEST_DATABASE_URL can come from the shell or from the git-ignored .env file
_env_file = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
TEST_URL = (
    os.environ.get("TEST_DATABASE_URL")
    or _env_file.get("TEST_DATABASE_URL")
    or "postgresql://postgres:postgres@localhost:5432/ai_news_test"
)
_db_name = TEST_URL.rsplit("/", 1)[-1].split("?")[0]
if "test" not in _db_name:
    raise RuntimeError(
        f"Refusing to run tests: database name {_db_name!r} does not contain 'test'. "
        "The tests wipe the schema they connect to. Set TEST_DATABASE_URL to a throwaway database."
    )

os.environ["DATABASE_URL"] = TEST_URL  # overrides any DATABASE_URL in the shell or .env
os.environ.setdefault("OPENAI_API_KEY", "test-key")