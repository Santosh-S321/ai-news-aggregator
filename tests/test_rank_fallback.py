from app.config import Settings
from app.llm import RankedItem
from app.rank import rerank_with_fallback


class StubLLM:
    def __init__(self, failing: set[str]):
        self.failing, self.calls = failing, []

    def rerank(self, profile_text, candidates, model=None):
        self.calls.append(model)
        if model in self.failing:
            raise RuntimeError("503 overloaded")
        return [RankedItem(id=1, score=7.0, reason="ok")]


def _settings():
    return Settings(llm_provider="gemini", gemini_api_key="x")


def test_falls_back_to_lighter_model():
    s = _settings()
    llm = StubLLM({s.rerank_model_name})
    out = rerank_with_fallback(llm, s, "profile", [{"id": 1}])
    assert out[0].score == 7.0
    assert llm.calls == [s.rerank_model_name, s.summary_model_name]


def test_returns_empty_when_all_models_fail():
    s = _settings()
    llm = StubLLM({s.rerank_model_name, s.summary_model_name})
    assert rerank_with_fallback(llm, s, "profile", [{"id": 1}]) == []