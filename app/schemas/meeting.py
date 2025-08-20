from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TranscriptionSegment(BaseModel):
    speaker: str
    text: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None

class MeetingStatusResponse(BaseModel):
    success: int = 0
    percentage: str
    bbhName: str
    Type: str
    Host: str
    transcription: Optional[List[TranscriptionSegment]] = []

class MeetingStatusData(BaseModel):
    status: int
    Data: MeetingStatusResponse

class SummaryResponse(BaseModel):
    success: int = 0
    requestId: str
    summary: str

class ChatRequest(BaseModel):
    requestId: str
    username: str
    message: str

class ChatResponse(BaseModel):
    success: int = 0
    response: str
