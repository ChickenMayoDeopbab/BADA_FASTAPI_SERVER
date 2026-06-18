from app.core.enums import ALL_DIFFICULTIES, ALL_PERSONALITIES, ScenarioCategory
from app.schemas.scenario import ScenarioInfo

PRESET_SCENARIOS: list[dict] = [
    {
        "scenario_id": 1,
        "category": ScenarioCategory.RESTAURANT,
        "title": "음식점 예약",
        "content": "레스토랑에 전화해 날짜·시간·인원을 말하고 자리를 예약합니다.",
        "call_target": "레스토랑 직원",
        "call_purpose": "음식점 예약",
        "ai_role": "레스토랑 예약 담당 직원",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are handling restaurant reservations over the phone. "
            "Confirm the customer's preferred date, time, and number of guests, "
            "then process the reservation naturally and professionally."
        ),
        "script": [
            {"step": 1, "ai_goal": "전화를 응대하고 예약 의사를 확인한다", "hint": "예약하고 싶다고 말해보세요"},
            {"step": 2, "ai_goal": "희망 날짜와 시간을 묻는다", "hint": "원하는 날짜와 시간을 말하세요"},
            {"step": 3, "ai_goal": "예약 인원수를 확인한다", "hint": "몇 명인지 말하세요"},
            {"step": 4, "ai_goal": "예약 내용을 확정하고 마무리한다", "hint": "예약 내용을 확인하고 인사하세요"},
        ],
        "difficulties": ALL_DIFFICULTIES,
        "personalities": ALL_PERSONALITIES,
        "is_custom": False,
    }, {
        "scenario_id": 2,
        "category": ScenarioCategory.RESTAURANT,
        "title": "음식점 예약 변경/취소",
        "content": "레스토랑에 전화해 기존 예약을 변경하거나 취소합니다.",
        "call_target": "레스토랑 직원",
        "call_purpose": "음식점 예약 변경 또는 취소",
        "ai_role": "레스토랑 예약 담당 직원",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are handling restaurant reservation changes and cancellations over the phone. "
            "If the customer wants to change the reservation, confirm the new date and time. "
            "If they want to cancel, ask for the reason politely before processing it."
        ),
        "script": [
            {"step": 1, "ai_goal": "전화를 응대하고 기존 예약을 확인한다", "hint": "예약을 변경/취소한다고 말하세요"},
            {"step": 2, "ai_goal": "예약자 이름이나 날짜로 예약을 조회한다", "hint": "예약자 이름과 날짜를 말하세요"},
            {"step": 3, "ai_goal": "변경 내용 또는 취소 사유를 확인한다", "hint": "바꿀 시간이나 취소 이유를 말하세요"},
            {"step": 4, "ai_goal": "처리 결과를 안내하고 마무리한다", "hint": "처리 내용을 확인하고 인사하세요"},
        ],
        "difficulties": ALL_DIFFICULTIES,
        "personalities": ALL_PERSONALITIES,
        "is_custom": False,
    }, {
        "scenario_id": 3,
        "category": ScenarioCategory.HOSPITAL,
        "title": "병원 진료 예약",
        "content": "내과·치과 등 병원에 전화해 원하는 날짜에 진료를 예약합니다.",
        "call_target": "병원 데스크 직원",
        "call_purpose": "병원 진료 예약",
        "ai_role": "병원 접수 데스크 직원",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are working at a hospital reception desk. "
            "Handle appointment requests by confirming the department, preferred date, and available time slots."
        ),
        "script": [
            {"step": 1, "ai_goal": "전화를 응대하고 용건을 확인한다", "hint": "진료 예약을 하고 싶다고 말하세요"},
            {"step": 2, "ai_goal": "진료과와 증상을 묻는다", "hint": "어느 과, 어떤 증상인지 말하세요"},
            {"step": 3, "ai_goal": "희망 날짜와 시간을 확인한다", "hint": "원하는 날짜와 시간을 말하세요"},
            {"step": 4, "ai_goal": "예약을 확정하고 준비물을 안내한다", "hint": "예약 내용을 확인하고 인사하세요"},
        ],
        "difficulties": ALL_DIFFICULTIES,
        "personalities": ALL_PERSONALITIES,
        "is_custom": False,
    }, {
        "scenario_id": 4,
        "category": ScenarioCategory.HOSPITAL,
        "title": "병원 진료 결과 문의",
        "content": "내과·치과 등 병원에 전화해 검사 결과나 처방전에 대해 문의합니다.",
        "call_target": "병원 간호사",
        "call_purpose": "병원 진료 결과 문의",
        "ai_role": "병원 간호사",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are a hospital nurse answering phone inquiries about medical test results and prescriptions. "
            "Provide general guidance naturally and recommend speaking with a doctor if necessary."
        ),
        "script": [
            {"step": 1, "ai_goal": "전화를 응대하고 문의 목적을 확인한다", "hint": "검사 결과를 문의한다고 말하세요"},
            {"step": 2, "ai_goal": "이름과 진료일로 본인 확인을 한다", "hint": "이름과 진료받은 날짜를 말하세요"},
            {"step": 3, "ai_goal": "검사 결과나 처방을 안내한다", "hint": "궁금한 점을 물어보세요"},
            {"step": 4, "ai_goal": "추가 안내 후 통화를 마무리한다", "hint": "안내를 확인하고 인사하세요"},
        ],
        "difficulties": ALL_DIFFICULTIES,
        "personalities": ALL_PERSONALITIES,
        "is_custom": False,
    }, {
        "scenario_id": 5,
        "category": ScenarioCategory.COMPLAINT,
        "title": "배달 지연 문의",
        "content": "배달이 늦어진 상황에서 고객센터에 전화해 해결을 요청합니다.",
        "call_target": "택배 고객센터 상담원",
        "call_purpose": "배달 지연 문의",
        "ai_role": "배달 서비스 고객센터 상담원",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are working at a delivery service customer support center. "
            "Listen to the customer's complaint about a delayed delivery, "
            "check the current status, and suggest realistic solutions."
        ),
        "script": [
            {"step": 1, "ai_goal": "전화를 응대하고 문의 내용을 확인한다", "hint": "배달이 늦는다고 말하세요"},
            {"step": 2, "ai_goal": "주문 번호나 정보를 확인한다", "hint": "주문 번호나 내용을 말하세요"},
            {"step": 3, "ai_goal": "배달 현황을 안내하고 사과한다", "hint": "언제 도착하는지 물어보세요"},
            {"step": 4, "ai_goal": "해결책을 제시하고 마무리한다", "hint": "원하는 조치를 말하고 인사하세요"},
        ],
        "difficulties": ALL_DIFFICULTIES,
        "personalities": ALL_PERSONALITIES,
        "is_custom": False,
    }, {
        "scenario_id": 6,
        "category": ScenarioCategory.COMPLAINT,
        "title": "제품 환불·교환 요청",
        "content": "구매한 제품에 문제가 있어 환불 또는 교환을 요청합니다.",
        "call_target": "쇼핑물 고객센터 상담원",
        "call_purpose": "제품 환불 및 교환 요청",
        "ai_role": "쇼핑몰 고객센터 상담원",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are a shopping mall customer support agent. "
            "Handle refund or exchange requests, explain the required process, "
            "and ask necessary questions about the product issue."
        ),
        "script": [
            {"step": 1, "ai_goal": "전화를 응대하고 문의 목적을 확인한다", "hint": "환불/교환을 하고 싶다고 말하세요"},
            {"step": 2, "ai_goal": "주문 정보와 제품 문제를 확인한다", "hint": "주문 내용과 문제를 말하세요"},
            {"step": 3, "ai_goal": "환불/교환 절차를 안내한다", "hint": "어떻게 진행되는지 물어보세요"},
            {"step": 4, "ai_goal": "처리 방법을 확정하고 마무리한다", "hint": "원하는 처리를 말하고 인사하세요"},
        ],
        "difficulties": ALL_DIFFICULTIES,
        "personalities": ALL_PERSONALITIES,
        "is_custom": False,
    }, {
        "scenario_id": 7,
        "category": ScenarioCategory.DELIVERY,
        "title": "택배 배송 조회 및 변경",
        "content": "택배 회사에 전화해 배송 현황을 확인하고 수령지를 변경합니다.",
        "call_target": "택배 고객센터 상담원",
        "call_purpose": "택배 배송 조회 및 변경",
        "ai_role": "택배 회사 고객센터 상담원",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are a courier company customer support agent. "
            "Answer questions about package delivery status and "
            "help the customer change the delivery address if possible."
        ),
        "script": [
            {"step": 1, "ai_goal": "전화를 응대하고 문의 내용을 확인한다", "hint": "택배 배송을 문의한다고 말하세요"},
            {"step": 2, "ai_goal": "송장 번호나 주문 정보를 확인한다", "hint": "송장 번호나 주문 정보를 말하세요"},
            {"step": 3, "ai_goal": "배송 현황을 안내한다", "hint": "지금 어디쯤인지 물어보세요"},
            {"step": 4, "ai_goal": "수령지 변경을 처리하고 마무리한다", "hint": "받을 주소를 바꾸고 싶다고 말하세요"},
        ],
        "difficulties": ALL_DIFFICULTIES,
        "personalities": ALL_PERSONALITIES,
        "is_custom": False,
    }, {
        "scenario_id": 8,
        "category": ScenarioCategory.BANK,
        "title": "은행 업무 문의",
        "content": "은행 고객센터에 전화해 계좌·이체·대출 관련 기본 문의를 합니다.",
        "call_target": "은행 직원",
        "call_purpose": "은행 업무 문의",
        "ai_role": "은행 고객센터 상담 직원",
        "scenario_image": None,
        "tts_voice_id": None,
        "ai_prompt": (
            "You are a bank customer service representative. "
            "Answer customer questions related to bank accounts, transfers, cards, "
            "and loans accurately and professionally."
        ),
        "script": [
            {"step": 1, "ai_goal": "전화를 응대하고 문의 목적을 확인한다", "hint": "은행 업무를 문의한다고 말하세요"},
            {"step": 2, "ai_goal": "본인 확인을 안내한다", "hint": "이름 등 본인 확인 정보를 말하세요"},
            {"step": 3, "ai_goal": "문의 내용을 안내한다", "hint": "궁금한 업무를 물어보세요"},
            {"step": 4, "ai_goal": "추가 안내 후 통화를 마무리한다", "hint": "안내를 확인하고 인사하세요"},
        ],
        "difficulties": ALL_DIFFICULTIES,
        "personalities": ALL_PERSONALITIES,
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
