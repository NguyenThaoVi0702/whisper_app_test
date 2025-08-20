import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL = f'postgresql+psycopg2://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}'
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    LITE_LLM_API_KEY: str = os.getenv("LITE_LLM_API_KEY")
    LITE_LLM_BASE_URL: str = os.getenv("LITE_LLM_BASE_URL")
    LITE_LLM_MODEL_NAME: str = os.getenv("LITE_LLM_MODEL_NAME")
    DIARIZATION_SERVICE_URL: str = os.getenv("DIARIZATION_SERVICE_URL")
    TEMP_STORAGE_PATH: str = os.getenv("TEMP_STORAGE_PATH", "./audio_chunks")
    LIMIT_TURN= int(os.getenv("LIMIT_TURN", 6))

settings = Settings()
