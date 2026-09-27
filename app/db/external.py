"""Spring이 소유한 테이블 읽기 정의

여기 정의된 Spring 테이블은 FastAPI 가 생성이나 변경 절대 금지
"""

from sqlalchemy import BigInteger, Column, DateTime, MetaData, SmallInteger, String, Table, text

external_metadata = MetaData()

users_table = Table(
    "users",
    external_metadata,
    Column("user_id", BigInteger, primary_key=True),
    Column("name", String(50)),
    Column("profile_image", String(255)),
    Column("role", String(20)),
    Column("status", String(20), nullable=False, server_default=text("'ACTIVE'")),
    Column("suspended_until", DateTime(timezone=True)),
)

# FileORM 과 같은 테이블. user_id 는 Spring 업로드가 채우는 칸이라 여기서만 읽는다.
files_table = Table(
    "file",
    external_metadata,
    Column("file_id", BigInteger, primary_key=True),
    Column("file_type", String(32)),
    Column("s3_key", String(512)),
    Column("title", String(255)),
    Column("user_id", BigInteger),
)

training_records_table = Table(
    "training_records",
    external_metadata,
    Column("record_id", BigInteger, primary_key=True),
    Column("user_id", BigInteger),
    Column("scenario_name", String(255)),
    Column("session_type", String(32)),
    Column("started_at", DateTime),
    Column("duration_seconds", BigInteger),
    Column("anxiety_score", SmallInteger),
    Column("recording_key", String(512)),
)
