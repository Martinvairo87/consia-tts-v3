import base64
import io
import os
from typing import Iterator

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from kokoro import KPipeline

LANG_CODE = os.getenv("KOKORO_LANG_CODE", "e")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "ef_dora")
DEFAULT_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))
SAMPLE_RATE = 24000

app = FastAPI(title="CONSIA TTS V3")

PIPELINE = KPipeline(lang_code=LANG_CODE)

class SynthesizeIn(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None

def ensure_np(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)

def text_to_visemes(text: str) -> list[str]:
    mapping = {
        "a": "A", "á": "A",
        "e": "E", "é": "E",
        "i": "I", "í": "I",
        "o": "O", "ó": "O",
        "u": "U", "ú": "U",
        "m": "M", "b": "M", "p": "M",
        "f": "F", "v": "F",
    }
    return [mapping.get(ch.lower(), "S") for ch in text]

def synthesize_audio(text: str, voice: str, speed: float) -> tuple[np.ndarray, list[str]]:
    generator = PIPELINE(
        text,
        voice=voice,
        speed=speed,
        split_pattern=r"\n+"
    )

    chunks: list[np.ndarray] = []
    for _, _, audio in generator:
        chunks.append(ensure_np(audio))

    if not chunks:
        return np.zeros((SAMPLE_RATE // 4,), dtype=np.float32), ["S"]

    full_audio = np.concatenate(chunks)
    visemes = text_to_visemes(text)
    return full_audio, visemes

def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()

def iter_chunks(data: bytes, chunk_size: int = 16384) -> Iterator[bytes]:
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "CONSIA TTS V3",
        "lang_code": LANG_CODE,
        "default_voice": DEFAULT_VOICE,
        "sample_rate": SAMPLE_RATE
    }

@app.post("/synthesize")
def synthesize(payload: SynthesizeIn):
    text = payload.text.strip()
    if not text:
        return JSONResponse({"ok": False, "error": "missing_text"}, status_code=400)

    voice = payload.voice or DEFAULT_VOICE
    speed = payload.speed or DEFAULT_SPEED

    audio, visemes = synthesize_audio(text, voice, speed)
    data = wav_bytes(audio)
    audio_b64 = base64.b64encode(data).decode("utf-8")

    return {
        "ok": True,
        "voice": voice,
        "speed": speed,
        "sample_rate": SAMPLE_RATE,
        "audio_base64": audio_b64,
        "audio_mime": "audio/wav",
        "visemes": visemes
    }

@app.post("/synthesize_stream")
def synthesize_stream(payload: SynthesizeIn):
    text = payload.text.strip()
    if not text:
        return JSONResponse({"ok": False, "error": "missing_text"}, status_code=400)

    voice = payload.voice or DEFAULT_VOICE
    speed = payload.speed or DEFAULT_SPEED

    audio, _ = synthesize_audio(text, voice, speed)
    data = wav_bytes(audio)

    return StreamingResponse(
        iter_chunks(data),
        media_type="audio/wav",
        headers={
            "X-Consia-Voice": voice,
            "X-Consia-Sample-Rate": str(SAMPLE_RATE)
        }
    )
