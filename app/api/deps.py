import logging
from typing import Generator

from fastapi import Depends, HTTPException, Form, Query, status
from sqlmodel import Session, select

from app.db.base import engine
from app.db.models import User, MeetingJob

logger = logging.getLogger(__name__)

# ===================================================================
#   DATABASE DEPENDENCIES
# ===================================================================

def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency that creates and yields a new database session for each request. 
    """
    with Session(engine) as session:
        yield session


# ===================================================================
#   USER & AUTHORIZATION DEPENDENCIES
# ===================================================================

def get_or_create_user(
    session: Session = Depends(get_db_session),
    username: str = Form(...)
) -> User:
    """
    Dependency to get a user by username from the DB.
    If the user does not exist, a new one is created.
    """
    user = session.exec(select(User).where(User.username == username)).first()
    
    if not user:
        logger.info(f"User '{username}' not found. Creating new user record.")
        user = User(username=username, display_name=username)
        session.add(user)
        session.commit()
        session.refresh(user)
        
    return user

def get_or_create_user_from_query(
    session: Session = Depends(get_db_session),
    username: str = Query(..., description="The username of the user making the request.")
) -> User:
    """
    Dependency to get a user by username from a URL Query Parameter.
    Used for GET requests.
    """
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username query parameter is required.")
        
    user = session.exec(select(User).where(User.username == username)).first()
    
    if not user:
        logger.info(f"User '{username}' not found from query. Creating new user record.")
        user = User(username=username, display_name=username)
        session.add(user)
        session.commit()
        session.refresh(user)
        
    return user


def get_and_verify_job_owner(
    current_user: User = Depends(get_or_create_user),
    requestId: str = Form(...),
    session: Session = Depends(get_db_session)
) -> MeetingJob:
    """
    Verifies that the user making the request is the actual owner of the job.
    """
    logger.debug(f"Verifying ownership for job '{requestId}' by user '{current_user.username}'")
    
    job = session.exec(select(MeetingJob).where(MeetingJob.request_id == requestId)).first()
    
    if not job:
        logger.warning(f"Access attempt failed: Meeting job with requestId '{requestId}' not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting job with requestId '{requestId}' not found."
        )
        
    if job.user_id != current_user.id:
        logger.warning(
            f"FORBIDDEN: User '{current_user.username}' (ID: {current_user.id}) "
            f"attempted to access job '{requestId}' owned by user ID '{job.user_id}'."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to access this resource."
        )
        
    return job
