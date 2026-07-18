# Use a lightweight official Python image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy all source code and config files
COPY . .

#  Ensure key credential files are accessible inside the container
ENV GMAIL_CREDENTIALS_FILE=/app/credentials.json
ENV GMAIL_TOKEN_FILE=/app/token.json
ENV CALENDAR_CREDENTIALS_FILE=/app/credentials.json
ENV CALENDAR_TOKEN_FILE=/app/calendar_token.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/autocorp_storage.json
ENV PYTHONUTF8=1

RUN mkdir -p logs chroma_data

VOLUME ["/app/chroma_data"]

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
