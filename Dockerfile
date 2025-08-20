# Stage 1
FROM tiangolo/uvicorn-gunicorn-fastapi:latest AS builder

COPY debian.sources /etc/apt/sources.list.d/debian.sources

RUN apt-get update && \
    apt-get install -y ffmpeg

WORKDIR /app

RUN pip config set global.index-url https://dso-nexus...

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ENV PATH="/app/venv/bin:$PATH"


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
