FROM dso-nexus.vietinbank.vn/ai_docker/finetune_whisper_lora:v2

RUN pip config set global.index-url https://dso-nexus.vietinbank.vn/repository/itd_pypi/simple
RUN pip config set global.trusted-host dso-nexus.vietinbank.vn

RUN pip install evaluate jiwer


WORKDIR /app
