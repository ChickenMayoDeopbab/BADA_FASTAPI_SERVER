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
        "example_dialogue": [
            {"speaker": "ai", "text": "네, 안녕하세요. 바다레스토랑입니다. 무엇을 도와드릴까요?"},
            {"speaker": "user", "text": "안녕하세요, 예약을 하고 싶어서 전화드렸어요."},
            {"speaker": "ai", "text": "네, 예약 도와드리겠습니다. 원하시는 날짜와 시간이 어떻게 되세요?"},
            {"speaker": "user", "text": "이번 주 토요일 저녁 6시에 가능할까요?"},
            {"speaker": "ai", "text": "네, 토요일 저녁 6시 가능합니다. 몇 분이 방문하실 예정인가요?"},
            {"speaker": "user", "text": "어른 4명이요."},
            {"speaker": "ai", "text": "네, 확인했습니다. 예약자 성함을 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "배준하입니다."},
            {"speaker": "ai", "text": "배준하 님, 토요일 저녁 6시에 4분 예약 완료됐습니다. 토요일에 뵙겠습니다."},
            {"speaker": "user", "text": "네, 감사합니다. 안녕히 계세요."},
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
        "example_dialogue": [
            {"speaker": "ai", "text": "네, 안녕하세요. 바다레스토랑입니다. 무엇을 도와드릴까요?"},
            {"speaker": "user", "text": "안녕하세요, 예약을 변경하고 싶어서 전화드렸어요."},
            {"speaker": "ai", "text": "네, 예약 변경 도와드리겠습니다. 예약자 성함과 예약 날짜를 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "조상철이고요, 이번 주 토요일 저녁 6시로 예약했어요."},
            {"speaker": "ai", "text": "확인됐습니다. 토요일 저녁 6시 4분 예약이 있으시네요. 어떻게 변경해 드릴까요?"},
            {"speaker": "user", "text": "일요일 저녁 7시로 바꿀 수 있을까요?"},
            {"speaker": "ai", "text": "네, 일요일 저녁 7시 가능합니다. 인원은 그대로 4분 맞으실까요?"},
            {"speaker": "user", "text": "네, 인원은 그대로예요."},
            {"speaker": "ai", "text": "네, 그럼 일요일 저녁 7시 4분으로 변경해 드렸습니다. 일요일에 뵙겠습니다."},
            {"speaker": "user", "text": "감사합니다. 안녕히 계세요."},
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
        "example_dialogue": [
            {"speaker": "ai", "text": "네, 안녕하세요. 바다내과입니다. 무엇을 도와드릴까요?"},
            {"speaker": "user", "text": "안녕하세요, 진료 예약을 하고 싶어서 전화드렸어요."},
            {"speaker": "ai", "text": "네, 어디가 불편하셔서 그러세요?"},
            {"speaker": "user", "text": "며칠 전부터 기침이 심하고 목이 아파서요."},
            {"speaker": "ai", "text": "네, 알겠습니다. 원하시는 날짜와 시간이 있으신가요?"},
            {"speaker": "user", "text": "내일 오전 10시쯤 가능할까요?"},
            {"speaker": "ai", "text": "네, 내일 오전 10시 예약 가능합니다. 성함과 생년월일을 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "변성우이고요, 2009년 12월 25일입니다."},
            {"speaker": "ai", "text": "변성우 님, 내일 오전 10시로 예약해 드렸습니다. 오실 때 신분증을 챙겨 오세요."},
            {"speaker": "user", "text": "네, 알겠습니다. 감사합니다."},
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
        "example_dialogue": [
            {"speaker": "ai", "text": "네, 바다내과입니다. 무엇을 도와드릴까요?"},
            {"speaker": "user", "text": "안녕하세요, 지난주에 받은 검사 결과를 문의하려고 전화드렸어요."},
            {"speaker": "ai", "text": "네, 본인 확인이 필요합니다. 성함과 진료받으신 날짜를 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "박준석이고요, 지난주 화요일에 혈액 검사를 받았어요."},
            {"speaker": "ai", "text": "네, 확인됐습니다. 검사 결과는 모두 정상 범위로 나왔습니다."},
            {"speaker": "user", "text": "다행이네요. 그럼 병원에 다시 가지 않아도 되나요?"},
            {"speaker": "ai", "text": "네, 증상이 없으시면 다시 안 오셔도 됩니다. 불편한 점이 생기면 내원해 주세요."},
            {"speaker": "user", "text": "네, 알겠습니다. 처방받은 약은 다 먹어야 하나요?"},
            {"speaker": "ai", "text": "네, 처방받으신 약은 남기지 말고 끝까지 드시는 게 좋습니다."},
            {"speaker": "user", "text": "네, 자세히 알려주셔서 감사합니다. 안녕히 계세요."},
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
        "example_dialogue": [
            {"speaker": "ai", "text": "네, 안녕하세요. 바다배달 고객센터입니다. 무엇을 도와드릴까요?"},
            {"speaker": "user", "text": "안녕하세요, 주문한 음식이 한 시간 넘게 안 와서 전화드렸어요."},
            {"speaker": "ai", "text": "불편을 드려 죄송합니다. 확인을 위해 주문 번호를 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "주문 번호는 1024번입니다."},
            {"speaker": "ai", "text": "확인해 보니 주문이 몰려 배달이 지연되고 있습니다. 기다리시게 해서 죄송합니다."},
            {"speaker": "user", "text": "그렇군요. 그럼 언제쯤 도착할까요?"},
            {"speaker": "ai", "text": "지금 기사님이 출발하셔서 15분 안에 도착할 예정입니다."},
            {"speaker": "user", "text": "알겠습니다. 그럼 조금만 더 기다려 볼게요."},
            {"speaker": "ai", "text": "네, 기다려 주셔서 감사합니다. 사과의 의미로 할인 쿠폰을 보내드리겠습니다."},
            {"speaker": "user", "text": "네, 감사합니다. 안녕히 계세요."},
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
        "example_dialogue": [
            {"speaker": "ai", "text": "네, 안녕하세요. 바다몰 고객센터입니다. 무엇을 도와드릴까요?"},
            {"speaker": "user", "text": "안녕하세요, 구매한 제품에 문제가 있어서 교환하고 싶어요."},
            {"speaker": "ai", "text": "네, 확인 도와드리겠습니다. 주문 번호와 어떤 문제인지 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "주문 번호는 5872번이고요, 어제 받은 티셔츠에 얼룩이 있어요."},
            {"speaker": "ai", "text": "불편을 드려 죄송합니다. 같은 상품으로 교환해 드릴까요, 아니면 환불을 원하세요?"},
            {"speaker": "user", "text": "같은 상품으로 교환하고 싶어요."},
            {"speaker": "ai", "text": "네, 기사님이 제품을 회수한 뒤 새 상품을 보내드립니다. 회수는 내일 가능합니다."},
            {"speaker": "user", "text": "네, 그럼 내일 회수 부탁드릴게요."},
            {"speaker": "ai", "text": "네, 접수 완료됐습니다. 새 상품은 회수 후 2~3일 안에 받아보실 수 있습니다."},
            {"speaker": "user", "text": "네, 감사합니다. 안녕히 계세요."},
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
        "example_dialogue": [
            {"speaker": "ai", "text": "네, 안녕하세요. 바다택배 고객센터입니다. 무엇을 도와드릴까요?"},
            {"speaker": "user", "text": "안녕하세요, 택배가 어디쯤 왔는지 확인하고 싶어서 전화드렸어요."},
            {"speaker": "ai", "text": "네, 조회 도와드리겠습니다. 송장 번호를 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "송장 번호는 654321입니다."},
            {"speaker": "ai", "text": "확인해 보니 지금 지역 물류센터에 도착해서 내일 오전 중 배송될 예정입니다."},
            {"speaker": "user", "text": "내일은 제가 집에 없는데, 받는 주소를 회사로 바꿀 수 있을까요?"},
            {"speaker": "ai", "text": "네, 가능합니다. 변경하실 주소를 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "대구광역시 동구 바다로 52, 바다빌딩 3층이요."},
            {"speaker": "ai", "text": "네, 말씀하신 주소로 변경해 드렸습니다. 내일 회사로 배송될 예정입니다."},
            {"speaker": "user", "text": "네, 감사합니다. 안녕히 계세요."},
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
        "example_dialogue": [
            {"speaker": "ai", "text": "네, 안녕하세요. 바다은행 고객센터입니다. 무엇을 도와드릴까요?"},
            {"speaker": "user", "text": "안녕하세요, 다른 은행으로 이체할 때 수수료가 궁금해서 전화드렸어요."},
            {"speaker": "ai", "text": "네, 안내드리겠습니다. 본인 확인을 위해 성함과 생년월일을 말씀해 주시겠어요?"},
            {"speaker": "user", "text": "조상철이고요, 2009년 5월 8일입니다."},
            {"speaker": "ai", "text": "확인됐습니다. 인터넷뱅킹 이체는 수수료가 무료이고, 창구는 건당 500원입니다."},
            {"speaker": "user", "text": "그렇군요. 그럼 하루에 이체할 수 있는 한도는 얼마인가요?"},
            {"speaker": "ai", "text": "계좌 이체 한도는 1회 1천만 원, 하루 5천만 원입니다. 더 궁금하신 점 있으세요?"},
            {"speaker": "user", "text": "아니요, 궁금한 건 다 여쭤봤어요. 감사합니다."},
            {"speaker": "ai", "text": "네, 이용해 주셔서 감사합니다. 좋은 하루 보내세요."},
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
