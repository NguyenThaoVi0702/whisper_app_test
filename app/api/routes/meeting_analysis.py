import json
import logging
from io import BytesIO

import requests
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.api.deps import get_db_session, get_and_verify_job_owner, get_or_create_user
from app.core.config import settings
from app.db.models import MeetingJob, Summary, User, ChatHistory
from app.schemas.meeting import ChatRequest, ChatResponse, SummaryResponse
from app.services.ai_service import ai_service
from app.utils import create_meeting_minutes_doc_buffer

logger = logging.getLogger(__name__)

router = APIRouter()

# ===================================================================
#   HELPER FUNCTIONS 
# ===================================================================

def get_or_create_summary(session: Session, job_id: int) -> Summary:
    """
    Retrieves a summary record for a given job ID.
    If it doesn't exist, a new one is created and returned.
    """
    summary = session.exec(
        select(Summary).where(Summary.meeting_job_id == job_id)
    ).first()

    if not summary:
        logger.info(f"No summary found for job_id {job_id}. Creating a new record.")
        summary = Summary(meeting_job_id=job_id)
        session.add(summary)
        session.commit()
        session.refresh(summary)
    
    return summary

def format_transcript_for_ai(transcript: list, by_speaker: bool = False) -> str:
    """Formats the final transcript into a string for the AI prompt."""
    if not transcript:
        return ""
    
    if by_speaker:
        return json.dumps(transcript, indent=2, ensure_ascii=False)
    else:
        return "\n".join([f"{item['speaker']}: {item['text']}" for item in transcript])

# ===================================================================
#   API ENDPOINTS
# ===================================================================

@router.post("/summarize-by-topic", response_model=SummaryResponse, summary="Generate meeting minutes (summary by topic)")
async def summarize_by_topic(
    job: MeetingJob = Depends(get_and_verify_job_owner),
    session: Session = Depends(get_db_session)
):
    """
    Generates or retrieves a formal meeting minutes document.
    """
    if job.status != "completed" or not job.final_transcript:
        raise HTTPException(status_code=400, detail="Meeting processing is not complete or transcript is missing.")

    db_summary = get_or_create_summary(session, job.id)

    if db_summary.summary_by_topic:
        logger.info(f"Topic summary for job {job.id} already exists. Returning cached version.")
        return SummaryResponse(requestId=job.request_id, summary=db_summary.summary_by_topic)

    logger.info(f"No existing summary found for job {job.id}. Requesting from AI service.")
    transcript_text = format_transcript_for_ai(job.final_transcript, by_speaker=False)

    meeting_info = {
        "bbh_name": job.bbh_name,
        "meeting_type": job.meeting_type,
        "meeting_host": job.meeting_host
    }
    
    try:
        summary_text = await ai_service.get_response(task="summarize_by_topic", user_message=transcript_text, meeting_info=meeting_info)

        if not summary_text or len(summary_text.split()) < 10:
            logger.error(f"AI returned an empty or unhelpful summary for job {job.id}.")
            raise HTTPException(status_code=502, detail="The AI service failed to generate a valid summary. Please try again.")

        db_summary.summary_by_topic = summary_text
        session.add(db_summary)
        session.commit()

        logger.info(f"Successfully generated and saved new topic summary for job {job.id}.")
        return SummaryResponse(requestId=job.request_id, summary=summary_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate topic summary for job {job.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating the summary: {str(e)}")


@router.post("/summarize-by-speaker", response_model=SummaryResponse, summary="Generate summary grouped by speaker")
async def summarize_by_speaker(
    job: MeetingJob = Depends(get_and_verify_job_owner),
    session: Session = Depends(get_db_session)
):
    """
    Generates or retrieves a summary of the meeting organized by speaker.
    """
    if job.status != "completed" or not job.final_transcript:
        raise HTTPException(status_code=400, detail="Meeting processing is not complete or transcript is missing.")

    db_summary = get_or_create_summary(session, job.id)

    if db_summary.summary_by_speaker:
        logger.info(f"Speaker summary for job {job.id} already exists. Returning cached version.")
        return SummaryResponse(requestId=job.request_id, summary=db_summary.summary_by_speaker)

    logger.info(f"No existing speaker summary found for job {job.id}. Requesting from AI service.")
    transcript_json_string = format_transcript_for_ai(job.final_transcript, by_speaker=True)

    meeting_info = {
        "bbh_name": job.bbh_name,
        "meeting_type": job.meeting_type,
        "meeting_host": job.meeting_host
    }

    try:
        summary_text = await ai_service.get_response(task="summarize_by_speaker", user_message=transcript_json_string, meeting_info=meeting_info)

        if not summary_text or len(summary_text.split()) < 10:
            logger.error(f"AI returned an empty or unhelpful speaker summary for job {job.id}.")
            raise HTTPException(status_code=502, detail="The AI service failed to generate a valid summary. Please try again.")

        db_summary.summary_by_speaker = summary_text
        session.add(db_summary)
        session.commit()

        logger.info(f"Successfully generated and saved new speaker summary for job {job.id}.")
        return SummaryResponse(requestId=job.request_id, summary=summary_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate speaker summary for job {job.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating the summary: {str(e)}")


@router.post("/get-conclusion", summary="Extract conclusion from the meeting summary")
async def get_conclusion(
    job: MeetingJob = Depends(get_and_verify_job_owner),
    session: Session = Depends(get_db_session)
):
    """
    Extracts the final conclusion and action items from the generated topic summary.
    """
    db_summary = get_or_create_summary(session, job.id)

    if not db_summary.summary_by_topic:
        raise HTTPException(status_code=400, detail="A topic summary must be generated first before extracting a conclusion.")

    try:
        logger.info(f"Requesting conclusion from AI for job_id: {job.id}")
        conclusion_text = await ai_service.get_response(
            task="get_conclusion", 
            user_message=db_summary.summary_by_topic
        )
        
        db_summary.conclusion = conclusion_text
        session.add(db_summary)
        session.commit()
        
        return {"status_code": 200, "conclusion": conclusion_text}

    except Exception as e:
        logger.error(f"Failed to get conclusion for job {job.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get conclusion: {str(e)}")


@router.post("/download-word", summary="Download meeting minutes as a .docx file")
async def download_word(
    job: MeetingJob = Depends(get_and_verify_job_owner),
    session: Session = Depends(get_db_session)
):
    """
    Downloads the main topic-based summary as a Word document.
    """
    logger.info(f"Request to download Word document for job_id: {job.id}")
    db_summary = get_or_create_summary(session, job.id)

    if not db_summary.summary_by_topic:
        raise HTTPException(status_code=404, detail="No summary available to download. Please generate a summary first.")

    try:
        buffer = create_meeting_minutes_doc_buffer(db_summary.summary_by_topic)
        
        filename = f"BienBanHop_{job.bbh_name.replace(' ', '_')}.docx"
        headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"} 
        
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers=headers
        )
    except Exception as e:
        logger.error(f"Error during DOCX generation for job {job.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during document generation.")


@router.post("/assign-tasks", summary="Send conclusion to n8n for task assignment")
async def assign_tasks(
    job: MeetingJob = Depends(get_and_verify_job_owner),
    session: Session = Depends(get_db_session)
):
    """
    Sends the extracted conclusion to an n8n webhook for further processing.
    """
    logger.info(f"Request to assign tasks for job_id: {job.id}")
    db_summary = get_or_create_summary(session, job.id)

    if not db_summary.conclusion:
        raise HTTPException(status_code=404, detail="No conclusion found for this meeting. Please generate a conclusion first.")

    try:
        webhook_url = "https://n8n-ai.vietinbank.vn/webhook/send-assign-tasks"
        payload = {"input_data": db_summary.conclusion}
        response = requests.post(webhook_url, json=payload, timeout=30)
        response.raise_for_status()
        
        logger.info(f"Successfully sent conclusion for job {job.id} to task assignment webhook.")
        return {"status_code": 200, "message": "Tasks successfully assigned."}

    except requests.RequestException as e:
        logger.error(f"Failed to send tasks to webhook for job {job.id}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to communicate with the task assignment service: {e}")


@router.post("/chat", response_model=ChatResponse, summary="Chat about the meeting content")
async def chat_with_meeting(
    req: ChatRequest,
    session: Session = Depends(get_db_session)
):
    """
    Handles conversational queries about a completed meeting, using the transcript,
    generated summary, and previous conversation turns as context.
    """
    logger.info(f"Chat request for requestId '{req.requestId}' from user '{req.username}'")

    current_user = get_or_create_user(session=session, username=req.username)
    job = session.exec(
        select(MeetingJob).where(MeetingJob.request_id == req.requestId)
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Meeting job with requestId '{req.requestId}' not found.")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden: You do not have permission to access this chat.")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Meeting is not yet complete. Please wait for the transcript.")
    
    db_summary = get_or_create_summary(session, job.id)
    if not db_summary.summary_by_topic:
        raise HTTPException(status_code=404, detail="No summary found. Please generate a summary first.")

    history_records = session.exec(
        select(ChatHistory)
        .where(ChatHistory.meeting_job_id == job.id)
        .order_by(ChatHistory.created_at.desc())
        .limit(settings.LIMIT_TURN)
    ).all()
    formatted_history = [{"role": record.role, "content": record.message} for record in reversed(history_records)]

    full_context_text = f"**Full Meeting Transcript:**\n{format_transcript_for_ai(job.final_transcript)}\n\n" \
                        f"**Meeting Summary:**\n{db_summary.summary_by_topic}"

    ai_user_message = f"Dựa vào nội dung cuộc họp sau đây, hãy trả lời câu hỏi của người dùng.\n\n" \
                      f"--- Bối cảnh cuộc họp ---\n{full_context_text}\n\n" \
                      f"--- Câu hỏi của người dùng ---\n{req.message}"
    
    try:
        ai_response = await ai_service.get_response(
            task="chat", 
            user_message=ai_user_message,
            history=formatted_history
        )
        
        user_chat = ChatHistory(meeting_job_id=job.id, role="user", message=req.message)
        assistant_chat = ChatHistory(meeting_job_id=job.id, role="assistant", message=ai_response)
        session.add(user_chat)
        session.add(assistant_chat)
        session.commit()

        return ChatResponse(success=0, response=ai_response)

    except Exception as e:
        logger.error(f"Chat failed for job {job.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed due to an internal error: {str(e)}")
