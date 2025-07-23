from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    message: str
    sample_rate: int
    file_id: str

class SpeechAnalysisResponse(BaseModel):
    speech: bool
    gender: Optional[str]
