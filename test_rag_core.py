import pytest
from rag_core import build_pipeline, ask_question, hybrid_search

@pytest.fixture(scope="module")
def pipeline():
    return build_pipeline()

def test_hybrid_search_returns_results(pipeline):
    results = hybrid_search(pipeline, "Harry's uncle job", n_results=5)
    assert len(results) == 5
    assert all("chunk" in r and "page" in r and "score" in r for r in results)

def test_ask_question_returns_answer_and_context(pipeline):
    answer, context = ask_question(pipeline, "What does Harry's uncle do for work?")
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert len(context) > 0

def test_ask_question_cites_page(pipeline):
    answer, _ = ask_question(pipeline, "What street do the Dursleys live on?")
    assert "Page" in answer or "page" in answer

def test_ask_question_handles_unanswerable(pipeline):
    answer, _ = ask_question(pipeline, "What is Dumbledores phone number?")
    refusal_phrases = [
        "don't know", "not mentioned", "cannot", "no information",
        "doesn't say", "does not", "not provide", "not available",
        "not stated", "not specified", "not include", "do not contain",
        "does not contain", "not contain"
    ]
    assert any(phrase in answer.lower() for phrase in refusal_phrases)