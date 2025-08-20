import os
import json
import shutil
import logging
import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException, BackgroundTasks
from sqlmodel import Session, select
from pydub import AudioSegment

from app.db.base import engine
from app.db.models import MeetingJob, User
from app.api.deps import get_db_session, get_or_create_user, get_and_verify_job_owner
from app.core.config import settings
from app.services.diarization_client import diarization_client
from app.schemas.meeting import MeetingStatusResponse, MeetingStatusData, TranscriptionSegment

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


logger = logging.getLogger(__name__)

router = APIRouter()


# ===================================================================
#   HELPER FUNCTIONS
# ===================================================================

def get_chunk_number(filename: str) -> int:
    """Extracts the chunk number from a filename like 'file_123.wav'."""
    match = re.search(r'_(\d+)\.\w+$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1

async def assemble_chunks(request_id: str) -> str:
    """
    Assembles sorted audio chunks into a single, standardized WAV audio file.
    """
    session_dir = Path(settings.TEMP_STORAGE_PATH) / request_id
    if not session_dir.is_dir():
        logger.error(f"Assembly failed: Directory not found for request_id '{request_id}'")
        raise FileNotFoundError(f"Directory for request_id '{request_id}' not found.")

    chunk_files = [f for f in os.listdir(session_dir) if get_chunk_number(f) != -1]
    if not chunk_files:
        logger.error(f"Assembly failed: No valid chunk files found in {session_dir}")
        raise FileNotFoundError(f"No valid chunk files found for request_id '{request_id}'.")

    chunk_files.sort(key=get_chunk_number)
    logger.info(f"Assembling {len(chunk_files)} chunks for request_id '{request_id}'...")

    combined_audio = AudioSegment.empty()
    for filename in chunk_files:
        chunk_path = session_dir / filename
        try:
            audio_chunk = AudioSegment.from_file(chunk_path)
            combined_audio += audio_chunk
        except CouldntDecodeError:
            logger.error(f"Could not decode file: '{chunk_path}'. It may be a corrupted or unsupported format.")
            raise IOError(f"Unsupported or corrupted audio chunk received: {filename}")
        except Exception as e:
            logger.error(f"Failed to load or combine chunk '{chunk_path}': {e}", exc_info=True)
            raise IOError(f"Could not process chunk file: {filename}")

    combined_audio = combined_audio.set_channels(1)
    combined_audio = combined_audio.set_frame_rate(16000)
    combined_audio = combined_audio.set_sample_width(2)

    with Session(engine) as session:
        job = session.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
        final_filename = f"{Path(job.original_filename).stem}_standardized.wav" if job else f"{request_id}_full_audio.wav"

    full_audio_path = session_dir / final_filename
    
    combined_audio.export(full_audio_path, format="wav")
    logger.info(f"Successfully assembled and standardized full audio at: {full_audio_path}")
    
    # Clean up the original chunk files
    for filename in chunk_files:
        try:
            os.remove(session_dir / filename)
        except OSError as e:
            logger.warning(f"Could not delete chunk file '{filename}': {e}")

    return str(full_audio_path)


async def process_full_audio(request_id: str):
    """
    Background task to assemble audio and trigger the diarization pipeline.
    """

    with Session(engine) as session:
        job = session.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
        if not job:
            logger.error(f"[BackgroundTask] Job with request_id '{request_id}' not found.")
            return

        try:
            standardized_audio_path = await assemble_chunks(request_id)
            language = job.language

            logger.info(f"Dispatching job '{request_id}' to diarization service with file '{standardized_audio_path}'...")
            diarization_response = await diarization_client.start_pipeline(standardized_audio_path, request_id, language=language)
            job.status = "processing"
            job.diarization_job_id = diarization_response.get("job_id")
            session.add(job)
            session.commit()
            logger.info(f"Job '{request_id}' successfully dispatched. Diarization Job ID: {job.diarization_job_id}")

        except Exception as e:
            error_message = f"Failed during background processing: {e}"
            logger.error(error_message, exc_info=True)
            job.status = "failed"
            job.error_message = error_message
            session.add(job)
            session.commit()


# ===================================================================
#   API ENDPOINTS
# ===================================================================

@router.post("/start-bbh", status_code=200, summary="Initialize a new meeting session")
async def start_bbh(
    session: Session = Depends(get_db_session),
    current_user: User = Depends(get_or_create_user),
    requestId: str = Form(...),
    language: str = Form(...),
    deviceData: str = Form(...),
    filename: str = Form(...),
    bbhName: str = Form(...),
    Type: str = Form(...),
    Host: str = Form(...),
):
    """
    Initializes a meeting job, creating a database record and a temporary directory to store incoming audio chunks.
    """
    logger.info(f"Initializing job for requestId: '{requestId}' by user '{current_user.username}'")

    try:
        device_info_dict = json.loads(deviceData)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in deviceData for requestId '{requestId}'. Storing raw string.")
        device_info_dict = {"raw": deviceData, "error": "Invalid JSON format"}

    # Create a dedicated directory for this session's chunks
    session_dir = Path(settings.TEMP_STORAGE_PATH) / requestId
    os.makedirs(session_dir, exist_ok=True)

    job = session.exec(select(MeetingJob).where(MeetingJob.request_id == requestId)).first()
    if job:
        logger.warning(f"Meeting job '{requestId}' already exists. Overwriting details.")
    else:
        job = MeetingJob(request_id=requestId)


    job.user_id = current_user.id
    job.language = language
    job.device_data = device_info_dict
    job.original_filename = filename
    job.bbh_name = bbhName
    job.meeting_type = Type
    job.meeting_host = Host
    job.status = "uploading"
    job.error_message = None
    job.final_transcript = None
    job.diarization_job_id = None

    session.add(job)
    session.commit()

    return {"status": 200, "message": "Meeting initialized successfully. Ready for chunk uploads."}


@router.post("/upload-file-chunk", status_code=202, summary="Upload a single audio chunk")
async def upload_file_chunk(
    background_tasks: BackgroundTasks,
    job: MeetingJob = Depends(get_and_verify_job_owner),
    session: Session = Depends(get_db_session),
    isLastChunk: bool = Form(...),
    FileData: UploadFile = File(...),
):
    """
    Receives an audio chunk, saves it, and if it's the last chunk,
    triggers a background task to assemble the audio and start the pipeline.
    """
    logger.info(f"Received chunk '{FileData.filename}' for requestId '{job.request_id}'. Is last: {isLastChunk}")

    session_dir = Path(settings.TEMP_STORAGE_PATH) / job.request_id
    chunk_path = session_dir / FileData.filename
    
    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(FileData.file, buffer)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chunk file: {e}")

    if isLastChunk:
        job.status = "assembling"
        session.add(job)
        session.commit()
        logger.info(f"Job '{job.request_id}' status updated to 'assembling'. Triggering background task.")
        
        background_tasks.add_task(process_full_audio, job.request_id)

    return {"status": 202, "message": f"Chunk '{FileData.filename}' received and accepted for processing."}


@router.get("/get-full-transcription", response_model=MeetingStatusData, summary="Poll for meeting transcription results")
async def get_full_transcription(
    job: MeetingJob = Depends(get_and_verify_job_owner),
    session: Session = Depends(get_db_session)
):
    """
    Polls for the status of a meeting job. 
    """
    logger.debug(f"Polling result for requestId '{job.request_id}' by user '{job.owner.username}'")

    response_data = MeetingStatusResponse(
        percentage="0",
        bbhName=job.bbh_name,
        Type=job.meeting_type,
        Host=job.meeting_host,
    )


    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error_message or "Unknown processing error.")
    
    if job.status == "completed":
        response_data.percentage = "100"
        response_data.transcription = [TranscriptionSegment(**item) for item in job.final_transcript] if job.final_transcript else []
        return MeetingStatusData(status=200, Data=response_data)

    if job.status == "uploading":
        response_data.percentage = "10"
        return MeetingStatusData(status=200, Data=response_data)
    if job.status == "assembling":
        response_data.percentage = "20"
        return MeetingStatusData(status=200, Data=response_data)

    if job.status == "processing" and job.diarization_job_id:
        try:
            logger.debug(f"Querying backend for diarization_job_id '{job.diarization_job_id}'")
            backend_result = await diarization_client.get_pipeline_results(job.diarization_job_id)
            backend_status = backend_result.get("status")

            if backend_status == "completed":
                final_transcript = backend_result.get("data")

                if not final_transcript or not isinstance(final_transcript, list):
                    logger.error(f"Diarization service returned empty or malformed transcript for job {job.diarization_job_id}.")
                    job.status = "failed"
                    job.error_message = "Processing completed, but the final transcript was empty or invalid."
                    session.add(job)
                    session.commit()
                    raise HTTPException(status_code=500, detail=job.error_message)

                job.status = "completed"
                job.final_transcript = backend_result.get("data")
                session.add(job)
                session.commit()
                response_data.percentage = "100"
                response_data.transcription = [TranscriptionSegment(**item) for item in job.final_transcript] if job.final_transcript else []
            
            elif backend_status == "failed":
                job.status = "failed"
                job.error_message = backend_result.get("error", "Unknown error from diarization service.")
                session.add(job)
                session.commit()
                raise HTTPException(status_code=500, detail=job.error_message)

            else:
                percentage_map = {"pending": 30, "processing": 40, "mapping": 60, "refining": 80}
                response_data.percentage = str(percentage_map.get(backend_status, 50))
            
            return MeetingStatusData(status=200, Data=response_data)

        except Exception as e:
            logger.error(f"Could not get status from diarization service for job '{job.diarization_job_id}': {e}", exc_info=True)
            raise HTTPException(status_code=502, detail="Error communicating with the backend processing service.")

    raise HTTPException(status_code=500, detail="Job is in an unknown or inconsistent state.")



@router.post("/cancel-meeting", summary="Cancel an ongoing meeting and clean up resources")
async def cancel_meeting(
    job: MeetingJob = Depends(get_and_verify_job_owner),
    session: Session = Depends(get_db_session)
):
    """
    Allows a user to cancel a meeting job that has been started but not completed.
    This will delete the database record and any uploaded audio chunks from the disk.
    """
    logger.info(f"Received cancellation request for job_id: {job.id} (requestId: {job.request_id})")

    if job.status not in ["uploading", "assembling"]:
        logger.warning(
            f"User '{job.owner.username}' attempted to cancel a job in a non-cancellable state: '{job.status}'"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Meeting cannot be cancelled. It is already in the '{job.status}' state."
        )


    session_dir = Path(settings.TEMP_STORAGE_PATH) / job.request_id
    if session_dir.exists() and session_dir.is_dir():
        try:
            shutil.rmtree(session_dir)
            logger.info(f"Successfully removed temporary directory: {session_dir}")
        except OSError as e:
            logger.error(f"Error removing directory {session_dir} during cancellation: {e}")

    try:
        session.delete(job)
        session.commit()
        logger.info(f"Successfully deleted MeetingJob record {job.id} from the database.")
    except Exception as e:
        logger.error(f"Error deleting job {job.id} from database during cancellation: {e}", exc_info=True)
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to clean up the meeting record from the database.")

    return {"status": 200, "message": "Meeting successfully cancelled and resources cleaned up."}
