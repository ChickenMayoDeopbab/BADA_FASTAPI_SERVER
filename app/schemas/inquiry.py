from pydantic import BaseModel, ConfigDict, Field


class InquiryCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(min_length=1, max_length=100, description="문의 제목")
    content: str = Field(min_length=1, max_length=3_000, description="문의 내용")


class InquiryResponse(BaseModel):
    message: str = "문의가 전송되었습니다."
