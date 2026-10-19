from app.core.preset_scenarios import PRESET_MAP
from app.services.example_service import _ns_d8

EXPECTED = {
    1: "bcec4745",
    2: "20af7c63",
    3: "85c9e51c",
    4: "d36ff548",
    5: "72e5a340",
    6: "5b777f73",
    7: "8c66b491",
    8: "68720b0d",
}


def test_preset_dialogues_unchanged() -> None:
    got = {sid: _ns_d8(preset["example_dialogue"]) for sid, preset in PRESET_MAP.items()}
    assert got == EXPECTED, "프리셋 대본이 바뀌었다 — 서버가 그 프리셋을 다시 굽는다, 배포 뒤 H 청취 후 이 표를 갱신"
