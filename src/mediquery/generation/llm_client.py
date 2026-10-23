
"""Claude API client for MediQuery answer generation."""

import logging

import anthropic
from anthropic import Anthropic

from mediquery.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    CLAUDE_TEMPERATURE,
)
from mediquery.generation.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Generate answers using the Anthropic Messages API."""

    def __init__(self, client: Anthropic | None = None) -> None:
        if client is not None:
            self.client = client
        else:
            if not ANTHROPIC_API_KEY:
                raise ValueError(
                    "La variable ANTHROPIC_API_KEY n'est pas configurée."
                )
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate(self, question: str, context: str) -> str:
        """Generate an answer grounded in the supplied context."""
        if not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        if not context.strip():
            raise ValueError("Le contexte ne peut pas être vide.")

        user_prompt = build_user_prompt(question, context)

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError:
            logger.exception("Échec de l'appel à l'API Anthropic.")
            raise

        answer_parts = [
            block.text
            for block in response.content
            if block.type == "text"
        ]
        answer = "\n".join(answer_parts).strip()

        if not answer:
            raise RuntimeError(
                "Claude a retourné une réponse textuelle vide."
            )

        return answer
