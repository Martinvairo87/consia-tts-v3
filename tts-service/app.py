from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io
import wave
import numpy as np

app = FastAPI()

def generate_voice(text: str):
    sample_rate = 22050
    duration = max(1, len(text) * 0.05)
    t = np.linspace(0, duration, int(sample_rate * duration))

    audio = 0.3 * np.sin(2 * np.pi * 220 * t)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    buffer.seek(0)
    return buffer

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/tts")
def tts(text: str = "CONSIA activo"):
    audio = generate_voice(text)
    return StreamingResponse(audio, media_type="audio/wav")
