"""Spring이 소유한 테이블 읽기 정의

여기 정의된 Spring 테이블은 FastAPI 가 생성이나 변경 절대 금지
"""

from sqlalchemy import BigInteger, Column, MetaData, String, Table

external_metadata = MetaData()

users_table = Table(
    "users",
    external_metadata,
    Column("user_id", BigInteger, primary_key=True),
    Column("name", String(50)),
    Column("profile_image", String(255)),
    # @Enumerated(EnumType.STRING) 이 남기는 enum 이름 — "USER" / "ADMIN"
    Column("role", String(20)),
)
