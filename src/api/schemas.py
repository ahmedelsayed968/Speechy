from pydantic import BaseModel, Field
from typing import Literal, Optional

class UploadResponse(BaseModel):
    message: str
    sample_rate: int
    file_id: str


class SpeechyModelResponse(BaseModel):
    label: Optional[Literal['male','female']] = None
    probability: Optional[float] = Field(default=None,le=1.0,ge=0.0)
    speech: bool
    message: Optional[str] = None