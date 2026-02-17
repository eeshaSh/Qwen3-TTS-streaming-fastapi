"""
Lightweight FastAPI server exposing an OpenAI-compatible /v1/audio/speech endpoint for streaming TTS audio.

- Accepts POST requests with JSON: {"input": "text to synthesize"}
- Streams PCM audio bytes as response (audio/wav)
- Each uvicorn process loads a single model instance (multi-process via nginx + supervisord)
- On-demand voice clone loading with per-process caching

Configuration via environment variables:
    TTS_NUM_WORKERS: Number of uvicorn worker processes (used by entrypoint.sh)
    TTS_VOICE_META_DIR: Directory for voice metadata JSON files (default: /app/voices)
    TTS_API_KEY: Optional API key for Bearer auth on /v1/* endpoints (unset = auth disabled)
"""

import asyncio
import hmac
import json
import os
import queue
import threading
import numpy as np
from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel
import time
import torch
from typing import Optional

_SENTINEL = object()


# ---------------------------------------------------------------------------
# App and model initialization — each uvicorn process gets its own model
# ---------------------------------------------------------------------------

app = FastAPI()

DEFAULT_VOICE_CLONE_REF_PATH = "eesha_voice_cloning.wav"
DEFAULT_TEXT = "Hello. This is an audio recording that's at least 5 seconds long. How are you doing today? Bye!"

VOICE_META_DIR = os.environ.get("TTS_VOICE_META_DIR", "/app/voices")
API_KEY = os.environ.get("TTS_API_KEY")
os.makedirs(VOICE_META_DIR, exist_ok=True)

print(f"[INIT] Loading model (PID={os.getpid()})...", flush=True)

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    use_cuda_graphs=False,
    compile_mode="max-autotune-no-cudagraphs",
    use_fast_codebook=True,
    compile_codebook_predictor=True,
    compile_talker=True,
)

# In-process voice clone cache: filepath → VoiceClonePromptItem
VOICE_CLONE_CACHE = {}

# Pre-build default voice at startup
VOICE_CLONE_CACHE[DEFAULT_VOICE_CLONE_REF_PATH] = model.create_voice_clone_prompt(
    ref_audio=DEFAULT_VOICE_CLONE_REF_PATH,
    ref_text=DEFAULT_TEXT,
)

# Semaphore: safety net when all workers are busy — queues cleanly instead of
# running two generations concurrently on the same model
generation_semaphore = asyncio.Semaphore(1)
_event_loop: Optional[asyncio.AbstractEventLoop] = None

print(f"[INIT] Model ready (PID={os.getpid()}).", flush=True)


# ---------------------------------------------------------------------------
# Auth dependency — enabled only when TTS_API_KEY is set
# ---------------------------------------------------------------------------

async def verify_api_key(request: Request):
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = auth[7:]
    if not hmac.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# On-demand voice loading
# ---------------------------------------------------------------------------

def _get_voice_clone_prompt(filepath: str):
    """Load voice clone prompt from cache or disk metadata. Returns None if not found."""
    if filepath in VOICE_CLONE_CACHE:
        return VOICE_CLONE_CACHE[filepath]

    basename = os.path.basename(filepath)
    meta_path = os.path.join(VOICE_META_DIR, f"{basename}.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    prompt = model.create_voice_clone_prompt(meta["ref_audio"], meta["ref_text"])
    VOICE_CLONE_CACHE[filepath] = prompt
    return prompt


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    input: str
    cloning_audio_filename: Optional[str] = None


class AddVoiceRequest(BaseModel):
    ref_audio_filename: str
    ref_text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/add_voice", dependencies=[Depends(verify_api_key)])
async def add_voice(body: AddVoiceRequest):
    """
    Register a voice for cloning. The .wav file must already exist in /app.
    Metadata is persisted to disk so other workers can load it on-demand.
    """
    filepath = f"/app/{body.ref_audio_filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

    try:
        # Persist metadata to disk for cross-worker discovery
        # Use basename to match _get_voice_clone_prompt's lookup key
        meta_path = os.path.join(VOICE_META_DIR, f"{os.path.basename(body.ref_audio_filename)}.json")
        with open(meta_path, "w") as f:
            json.dump({"ref_audio": filepath, "ref_text": body.ref_text}, f)

        # Eagerly cache in THIS worker
        prompt = model.create_voice_clone_prompt(filepath, body.ref_text)
        VOICE_CLONE_CACHE[filepath] = prompt

        return {"status": "success", "message": f"Voice added: {body.ref_audio_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def speech_endpoint(request: Request, body: SpeechRequest):
    global _event_loop
    if _event_loop is None:
        _event_loop = asyncio.get_running_loop()

    text = body.input
    language = "English"

    voice_clone_prompt = VOICE_CLONE_CACHE[DEFAULT_VOICE_CLONE_REF_PATH]
    if body.cloning_audio_filename:
        print(f"[INFO] Using cloning audio: {body.cloning_audio_filename}", flush=True)
        loaded = _get_voice_clone_prompt(body.cloning_audio_filename)
        if loaded is not None:
            voice_clone_prompt = loaded

    start = time.time()
    await generation_semaphore.acquire()
    print(
        f"[REQ] /v1/audio/speech PID={os.getpid()} input: {text[:60]}",
        flush=True,
    )

    def audio_stream():
        q = queue.Queue(maxsize=16)
        stop_event = threading.Event()

        def producer():
            ttfb_printed = False
            try:
                for chunk, sr in model.stream_generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    emit_every_frames=4,
                    decode_window_frames=80,
                    overlap_samples=512,
                ):
                    if stop_event.is_set():
                        print(
                            f"[CANCEL] PID={os.getpid()} client disconnected, aborting generation: {text[:60]}",
                            flush=True,
                        )
                        break
                    if not ttfb_printed:
                        ttfb = time.time() - start
                        print(
                            f"[TTFB] PID={os.getpid()} {ttfb:.3f}s input: {text[:60]}",
                            flush=True,
                        )
                        ttfb_printed = True
                    chunk_int16 = np.clip(chunk, -1.0, 1.0)
                    chunk_int16 = (chunk_int16 * 32767.0).astype(np.int16).tobytes()
                    try:
                        q.put(chunk_int16, timeout=5)
                    except queue.Full:
                        print(
                            f"[CANCEL] PID={os.getpid()} queue full (consumer gone), aborting: {text[:60]}",
                            flush=True,
                        )
                        break
            except Exception as e:
                print(f"[ERROR] PID={os.getpid()} Exception: {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                gen_time = time.time() - start
                print(
                    f"[GEN_DONE] PID={os.getpid()} {gen_time:.3f}s input: {text[:60]}",
                    flush=True,
                )
                _event_loop.call_soon_threadsafe(generation_semaphore.release)
                try:
                    q.put(_SENTINEL, timeout=1)
                except queue.Full:
                    pass

        threading.Thread(target=producer, daemon=True).start()

        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            stop_event.set()

    headers = {
        "Content-Type": "audio/L16; rate=24000",
        "Transfer-Encoding": "chunked",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Content-Type-Options": "nosniff",
    }
    return StreamingResponse(audio_stream(), headers=headers, media_type="audio/L16")


@app.get("/")
def root():
    return {
        "message": "Qwen3-TTS Streaming FastAPI server. POST /v1/audio/speech /v1/add_voice",
        "pid": os.getpid(),
    }
