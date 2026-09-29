FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion/ ./ingestion/
COPY retrieval/ ./retrieval/
COPY graph/ ./graph/
COPY llm/ ./llm/
COPY data/ ./data/
COPY main.py .

CMD ["python", "main.py"]