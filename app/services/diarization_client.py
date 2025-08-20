import httpx
from app.core.config import settings

class DiarizationServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def start_pipeline(self, audio_path: str, session_id: str, language: str) -> dict:
        with open(audio_path, 'rb') as audio_file:
            files = {'file': (audio_path.split('/')[-1], audio_file, 'audio/wav')}
            data = {'session_id': session_id, 'language': language}
            response = await self.client.post("/pipeline/process", files=files, data=data)
            response.raise_for_status()
            return response.json()

    async def get_pipeline_results(self, job_id: str) -> dict:
        response = await self.client.get(f"/pipeline/results/{job_id}")
        response.raise_for_status()
        return response.json()
        
diarization_client = DiarizationServiceClient(base_url=settings.DIARIZATION_SERVICE_URL)
