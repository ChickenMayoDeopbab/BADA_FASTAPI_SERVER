from app.core.preset_scenarios import PRESET_SCENARIOS


def test_all_presets_have_example_dialogue() -> None:
    for scenario in PRESET_SCENARIOS:
        dialogue = scenario.get("example_dialogue")
        assert isinstance(dialogue, list), f"{scenario['title']}: example_dialogue 없음"
        assert 8 <= len(dialogue) <= 12, f"{scenario['title']}: 턴 수 {len(dialogue)}"


def test_example_dialogue_turn_shape() -> None:
    for scenario in PRESET_SCENARIOS:
        for turn in scenario["example_dialogue"]:
            assert set(turn.keys()) == {"speaker", "text"}, scenario["title"]
            assert turn["speaker"] in ("ai", "user"), scenario["title"]
            assert isinstance(turn["text"], str) and turn["text"].strip(), scenario["title"]


def test_example_dialogue_starts_with_ai_and_alternates() -> None:
    """전화 응대 특성상 상대방(ai)이 먼저 받고, 화자는 교대로 말한다."""
    for scenario in PRESET_SCENARIOS:
        dialogue = scenario["example_dialogue"]
        assert dialogue[0]["speaker"] == "ai", scenario["title"]
        for prev, cur in zip(dialogue, dialogue[1:], strict=False):
            assert prev["speaker"] != cur["speaker"], f"{scenario['title']}: 연속 동일 화자"
