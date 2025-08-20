from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class GenericSuccessResponse(BaseModel):
    success: int = 0
    message: str

class SpeakerSearchItem(BaseModel):
    display_name: str
    user_ad: str

class SpeakerSearchResponse(BaseModel):
    success: int = 0
    message: str = "Search successful"
    Data: List[SpeakerSearchItem]

class SpeakerProfileData(BaseModel):
    qdrant_point_id: str
    payload: Dict[str, Any]
    has_vector: bool

class SpeakerProfileResponse(BaseModel):
    success: int = 0
    message: str = "Profile retrieved successfully"
    Data: SpeakerProfileData

class SpeakerProfileInfo(BaseModel):
    display_name: str
    user_ad: str
    enrolled_at_utc: Optional[str] = None
    num_enrollment_samples: Optional[int] = None

class AllSpeakersResponse(BaseModel):
    success: int = 0
    message: str = "Successfully retrieved all speaker profiles."
    Data: List[SpeakerProfileInfo]
