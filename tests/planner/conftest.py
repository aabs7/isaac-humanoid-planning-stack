"""Fixtures for the planner tests. (``smap`` and ``env`` come from tests/conftest.py.)"""

from __future__ import annotations

import pytest

from g1sim.task.llm import OllamaChat
from g1sim.task.planner import Planner


@pytest.fixture
def planner():
    """A planner for testing the pure helpers that never reach the model."""
    return Planner(llm=object(), verbose=False)


@pytest.fixture(scope="session")
def llm():
    """The real local model. Tests using this are marked ``llm`` and skip cleanly
    when Ollama is not running, so the default suite still works anywhere."""
    client = OllamaChat()
    if not client.available():
        pytest.skip(f"Ollama model '{client.model}' not reachable at {client.base_url}"
                    f" -- run `ollama serve` and `ollama pull {client.model}`")
    return client
