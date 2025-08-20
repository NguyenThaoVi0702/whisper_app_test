import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Session, Column, Relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    display_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    meeting_jobs: List["MeetingJob"] = Relationship(back_populates="owner")
    action_logs: List["SpeakerActionLog"] = Relationship(back_populates="submitter")

class MeetingJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: str = Field(unique=True, index=True)
    user_id: int = Field(foreign_key="user.id")
    original_filename: str
    bbh_name: str
    meeting_type: str
    meeting_host: str
    language: str = Field(default="vi")
    device_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    status: str = Field(default="uploading", index=True)
    diarization_job_id: Optional[str] = Field(default=None, index=True)
    final_transcript: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSONB))
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": func.now()})

    owner: User = Relationship(back_populates="meeting_jobs")
    summary: Optional["Summary"] = Relationship(back_populates="meeting_job", sa_relationship_kwargs={"cascade": "all, delete"})
    chat_history: List["ChatHistory"] = Relationship(back_populates="meeting_job", sa_relationship_kwargs={"cascade": "all, delete"})

class Summary(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_job_id: int = Field(foreign_key="meetingjob.id", unique=True)
    summary_by_topic: Optional[str] = None
    summary_by_speaker: Optional[str] = None
    conclusion: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": func.now()})

    meeting_job: MeetingJob = Relationship(back_populates="summary")

class SpeakerActionLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: str = Field(index=True)
    submitter_id: int = Field(foreign_key="user.id")
    action_type: str = Field(index=True)
    target_user_ad: Optional[str] = Field(default=None, index=True)
    device_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    payload: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    status: str
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    submitter: User = Relationship(back_populates="action_logs")


class ChatHistory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_job_id: int = Field(foreign_key="meetingjob.id")
    role: str  # "user" or "assistant"
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    meeting_job: "MeetingJob" = Relationship(back_populates="chat_history")


