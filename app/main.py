from fastapi import FastAPI, BackgroundTasks
from sqlmodel import SQLModel, Session, select
from app.api.routes import meeting_workflow, meeting_analysis, speaker_management
from app.db.base import engine
from app.db.models import MeetingJob
from app.core.config import settings
import os
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def cleanup_old_jobs():
    """Removes temporary directories for jobs that are old and incomplete."""
    logger.info("Running cleanup task for old, incomplete jobs...")
    threshold = datetime.utcnow() - timedelta(days=2) # Delete incomplete jobs older than 2 days
    temp_dir = Path(settings.TEMP_STORAGE_PATH)
    
    with Session(engine) as session:
        incomplete_jobs = session.exec(
            select(MeetingJob).where(
                MeetingJob.status.in_(["uploading", "assembling"]),
                MeetingJob.created_at < threshold
            )
        ).all()
        
        for job in incomplete_jobs:
            job_dir = temp_dir / job.request_id
            if job_dir.exists():
                logger.warning(f"Removing stale job directory: {job_dir}")
                shutil.rmtree(job_dir, ignore_errors=True)
                job.status = "failed"
                job.error_message = "Job timed out and was cleaned up automatically."
                session.add(job)
        session.commit()

app = FastAPI(title="VietinBank AI Meeting Assistant")

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)
    os.makedirs(settings.TEMP_STORAGE_PATH, exist_ok=True)
    cleanup_old_jobs()

app.include_router(meeting_workflow.router, prefix="/api/v1/meeting", tags=["Meeting Workflow"])
app.include_router(meeting_analysis.router, prefix="/api/v1/analysis", tags=["Meeting Analysis"])
app.include_router(speaker_management.router, prefix="/api/v1/speaker", tags=["Speaker Management"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
