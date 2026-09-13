import pytest


@pytest.fixture(autouse=True)
def _forget_learned_thinking_off():
    from app.services import llm

    llm._learned_off.clear()
    yield
    llm._learned_off.clear()
