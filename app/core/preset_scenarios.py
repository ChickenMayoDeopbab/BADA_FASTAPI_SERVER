from app.core.enums import Difficulty, Personality, ScenarioCategory
from app.schemas.scenario import ScenarioInfo

PRESET_SCENARIOS: list[dict] = [
    {
        "scenario_id": 1,
        "category": ScenarioCategory.RESTAURANT,
        "title": "음식점 예약",
        "content": "레스토랑에 전화해 날짜·시간·인원을 말하고 자리를 예약합니다.",
        "call_target": "레스토랑 직원",
        "call_purpose": "음식점 예약",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are handling restaurant reservations over the phone. "
            "Confirm the customer's preferred date, time, and number of guests, "
            "then process the reservation naturally and professionally."
        ),
        "difficulties": [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
        "personalities": [Personality.KIND, Personality.NEUTRAL, Personality.TOUGH, Personality.RUDE],
        "is_custom": False,
    }, {
        "scenario_id": 2,
        "category": ScenarioCategory.RESTAURANT,
        "title": "음식점 예약 변경/취소",
        "content": "레스토랑에 전화해 기존 예약을 변경하거나 취소합니다.",
        "call_target": "레스토랑 직원",
        "call_purpose": "음식점 예약 변경 또는 취소",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are handling restaurant reservation changes and cancellations over the phone. "
            "If the customer wants to change the reservation, confirm the new date and time. "
            "If they want to cancel, ask for the reason politely before processing it."
        ),
        "difficulties": [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
        "personalities": [Personality.KIND, Personality.NEUTRAL, Personality.TOUGH, Personality.RUDE],
        "is_custom": False,
    }, {
        "scenario_id": 3,
        "category": ScenarioCategory.HOSPITAL,
        "title": "병원 진료 예약",
        "content": "내과·치과 등 병원에 전화해 원하는 날짜에 진료를 예약합니다.",
        "call_target": "병원 데스크 직원",
        "call_purpose": "병원 진료 예약",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are working at a hospital reception desk. "
            "Handle appointment requests by confirming the department, preferred date, and available time slots."
        ),
        "difficulties": [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
        "personalities": [Personality.KIND, Personality.NEUTRAL, Personality.TOUGH, Personality.RUDE],
        "is_custom": False,
    }, {
        "scenario_id": 4,
        "category": ScenarioCategory.HOSPITAL,
        "title": "병원 진료 결과 문의",
        "content": "내과·치과 등 병원에 전화해 검사 결과나 처방전에 대해 문의합니다.",
        "call_target": "병원 간호사",
        "call_purpose": "병원 진료 결과 문의",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are a hospital nurse answering phone inquiries about medical test results and prescriptions. "
            "Provide general guidance naturally and recommend speaking with a doctor if necessary."
        ),
        "difficulties": [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
        "personalities": [Personality.KIND, Personality.NEUTRAL, Personality.TOUGH, Personality.RUDE],
        "is_custom": False,
    }, {
        "scenario_id": 5,
        "category": ScenarioCategory.COMPLAINT,
        "title": "배달 지연 문의",
        "content": "배달이 늦어진 상황에서 고객센터에 전화해 해결을 요청합니다.",
        "call_target": "택배 고객센터 상담원",
        "call_purpose": "배달 지연 문의",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are working at a delivery service customer support center. "
            "Listen to the customer's complaint about a delayed delivery, "
            "check the current status, and suggest realistic solutions."
        ),
        "difficulties": [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
        "personalities": [Personality.KIND, Personality.NEUTRAL, Personality.TOUGH, Personality.RUDE],
        "is_custom": False,
    }, {
        "scenario_id": 6,
        "category": ScenarioCategory.COMPLAINT,
        "title": "제품 환불·교환 요청",
        "content": "구매한 제품에 문제가 있어 환불 또는 교환을 요청합니다.",
        "call_target": "쇼핑물 고객센터 상담원",
        "call_purpose": "제품 환불 및 교환 요청",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are a shopping mall customer support agent. "
            "Handle refund or exchange requests, explain the required process, "
            "and ask necessary questions about the product issue."
        ),
        "difficulties": [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
        "personalities": [Personality.KIND, Personality.NEUTRAL, Personality.TOUGH, Personality.RUDE],
        "is_custom": False,
    }, {
        "scenario_id": 7,
        "category": ScenarioCategory.DELIVERY,
        "title": "택배 배송 조회 및 변경",
        "content": "택배 회사에 전화해 배송 현황을 확인하고 수령지를 변경합니다.",
        "call_target": "택배 고객센터 상담원",
        "call_purpose": "택배 배송 조회 및 변경",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are a courier company customer support agent. "
            "Answer questions about package delivery status and "
            "help the customer change the delivery address if possible."
        ),
        "difficulties": [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
        "personalities": [Personality.KIND, Personality.NEUTRAL, Personality.TOUGH, Personality.RUDE],
        "is_custom": False,
    }, {
        "scenario_id": 8,
        "category": ScenarioCategory.BANK,
        "title": "은행 업무 문의",
        "content": "은행 고객센터에 전화해 계좌·이체·대출 관련 기본 문의를 합니다.",
        "call_target": "은행 직원",
        "call_purpose": "은행 업무 문의",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are a bank customer service representative. "
            "Answer customer questions related to bank accounts, transfers, cards, "
            "and loans accurately and professionally."
        ),
        "difficulties": [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
        "personalities": [Personality.KIND, Personality.NEUTRAL, Personality.TOUGH, Personality.RUDE],
        "is_custom": False,
    }
]

PRESET_MAP: dict[int, dict] = {s["scenario_id"]: s for s in PRESET_SCENARIOS}

def scenario_to_info(scenario: dict) -> ScenarioInfo:
    return ScenarioInfo(
        scenario_id=scenario["scenario_id"],
        category=scenario["category"],
        title=scenario["title"],
        content=scenario["content"],
        scenario_image=scenario["scenario_image"],
        tts_voice_id=scenario["tts_voice_id"],
        ai_prompt=scenario["ai_prompt"],
        is_custom=scenario["is_custom"],
        difficulties=scenario["difficulties"],
        personalities=scenario["personalities"],
    )
