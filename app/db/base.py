from sqlmodel import create_engine
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, echo=False)

# ensure all models are registered before creating the tables.

from . import models
