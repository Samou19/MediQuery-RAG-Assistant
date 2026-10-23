
"""Unit tests for the Claude API client."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mediquery.generation.llm_client import ClaudeClient


def test_generate_returns_text_from_claude() -> None:
    """The client should return the generated text."""
    mock_client = Mock()
    mock_client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Réponse médicale de test.")
        ]
    )

    client = ClaudeClient(client=mock_client)
    answer = client.generate(
        question="Qu'est-ce que le diabète de type 2 ?",
        context="Page 12 : définition du diabète de type 2.",
    )

    assert answer == "Réponse médicale de test."
    mock_client.messages.create.assert_called_once()


def test_generate_rejects_empty_question() -> None:
    """The client should reject an empty question."""
    client = ClaudeClient(client=Mock())

    with pytest.raises(ValueError, match="question ne peut pas être vide"):
        client.generate("   ", "Contexte médical.")


def test_generate_rejects_empty_context() -> None:
    """The client should reject an empty context."""
    client = ClaudeClient(client=Mock())

    with pytest.raises(ValueError, match="contexte ne peut pas être vide"):
        client.generate("Question médicale ?", "   ")


def test_generate_rejects_empty_response() -> None:
    """The client should reject a response without text."""
    mock_client = Mock()
    mock_client.messages.create.return_value = SimpleNamespace(content=[])

    client = ClaudeClient(client=mock_client)

    with pytest.raises(RuntimeError, match="réponse textuelle vide"):
        client.generate("Question médicale ?", "Contexte médical.")
