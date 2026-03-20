FROM python:3.11-slim

WORKDIR /app

COPY tts-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tts-service/app.py .

EXPOSE 8010

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
