
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag_engine import answer_question, ABSTENTION_MESSAGE


class FakeVectorStore:
    """A simple substitute for FAISS during unit tests."""

    def __init__(self, documents):
        self.documents = documents
        self.last_question = None
        self.last_k = None

    def similarity_search(self, question, k=3):
        self.last_question = question
        self.last_k = k
        return self.documents[:k]


class FakeLLM:
    """A fake LLM that avoids real API calls."""

    def __init__(self, answer="A test answer."):
        self.answer = answer
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(content=self.answer)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content=(
                "The Transformer relies entirely on "
                "attention mechanisms."
            ),
            metadata={
                "source": "paper.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content=(
                "The architecture uses an encoder and decoder."
            ),
            metadata={
                "source": "paper.pdf",
                "page": 3,
            },
        ),
    ]


def test_empty_question_is_rejected():
    store = FakeVectorStore([])
    llm = FakeLLM()

    with pytest.raises(ValueError, match="cannot be empty"):
        answer_question(store, "   ", llm=llm)


def test_invalid_k_is_rejected():
    store = FakeVectorStore([])
    llm = FakeLLM()

    with pytest.raises(ValueError, match="at least 1"):
        answer_question(store, "What is attention?", llm=llm, k=0)


def test_answer_includes_sources(sample_documents):
    store = FakeVectorStore(sample_documents)
    llm = FakeLLM("The paper introduces an attention-based model.")

    result = answer_question(
        store,
        "What is the main contribution?",
        llm=llm,
    )

    assert result["answer"] == (
        "The paper introduces an attention-based model."
    )
    assert len(result["sources"]) == 2
    assert result["sources"][0] == {
        "source": "paper.pdf",
        "page": 1,
    }
    assert len(result["retrieved_passages"]) == 2


def test_retrieval_uses_requested_top_k(sample_documents):
    store = FakeVectorStore(sample_documents)
    llm = FakeLLM()

    answer_question(
        store,
        "What does the paper propose?",
        llm=llm,
        k=1,
    )

    assert store.last_question == "What does the paper propose?"
    assert store.last_k == 1


def test_empty_retrieval_returns_abstention():
    store = FakeVectorStore([])
    llm = FakeLLM()

    result = answer_question(
        store,
        "What is the paper about?",
        llm=llm,
    )

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["sources"] == []
    assert result["retrieved_passages"] == []
    assert llm.last_prompt is None
