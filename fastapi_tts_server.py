"""
Lightweight FastAPI server exposing an OpenAI-compatible /v1/audio/speech endpoint for streaming TTS audio.

- Accepts POST requests with JSON: {"input": "text to synthesize"}
- Streams PCM audio bytes as response (audio/wav)
- Loads multiple Qwen3TTSModel replicas at startup for concurrent request handling

Configuration via environment variables:
    TTS_NUM_REPLICAS: Number of model replicas to load (default: 4)
    TTS_QUEUE_TIMEOUT: Seconds to wait for a free replica before returning 503 (default: 30)
"""

import asyncio
import os
import numpy as np
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import time
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Replica pool — routes requests to independent model instances
# ---------------------------------------------------------------------------

@dataclass
class Replica:
    model: Qwen3TTSModel
    voice_clone_cache: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    index: int = 0


class ReplicaPool:
    def __init__(self, replicas: list[Replica], timeout: float = 30.0):
        self.replicas = replicas
        self.timeout = timeout
        self._available = asyncio.Event()
        self._available.set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def acquire(self) -> Replica:
        """Acquire a free replica, waiting up to self.timeout seconds."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        deadline = self._loop.time() + self.timeout
        while True:
            for replica in self.replicas:
                if not replica.lock.locked():
                    await replica.lock.acquire()
                    return replica

            remaining = deadline - self._loop.time()
            if remaining <= 0:
                raise HTTPException(
                    status_code=503,
                    detail="All model replicas are busy. Try again later.",
                )
            self._available.clear()
            try:
                await asyncio.wait_for(self._available.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=503,
                    detail="All model replicas are busy. Try again later.",
                )

    def release(self, replica: Replica):
        """Release a replica back to the pool. Thread-safe — can be called from
        the threadpool threads where sync generators run."""
        self._loop.call_soon_threadsafe(self._do_release, replica)

    def _do_release(self, replica: Replica):
        """Runs on the event loop thread where asyncio primitives are safe."""
        replica.lock.release()
        self._available.set()


# ---------------------------------------------------------------------------
# App and model initialization
# ---------------------------------------------------------------------------

app = FastAPI()

NUM_REPLICAS = int(os.environ.get("TTS_NUM_REPLICAS", "4"))
QUEUE_TIMEOUT = float(os.environ.get("TTS_QUEUE_TIMEOUT", "30"))

DEFAULT_VOICE_CLONE_REF_PATH = "eesha_voice_cloning.wav"
DEFAULT_TEXT = "Hello. This is an audio recording that's at least 5 seconds long. How are you doing today? Bye!"

print(f"[INIT] Loading {NUM_REPLICAS} model replica(s)...", flush=True)

_replicas: list[Replica] = []
for _i in range(NUM_REPLICAS):
    print(f"[INIT] Loading replica {_i}...", flush=True)
    _model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    _model.enable_streaming_optimizations(
        decode_window_frames=80,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="max-autotune-no-cudagraphs",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )
    _cache = {}
    _prompt = _model.create_voice_clone_prompt(
        ref_audio=DEFAULT_VOICE_CLONE_REF_PATH,
        ref_text=DEFAULT_TEXT,
    )
    _cache[DEFAULT_VOICE_CLONE_REF_PATH] = _prompt
    _replicas.append(Replica(model=_model, voice_clone_cache=_cache, index=_i))
    print(f"[INIT] Replica {_i} ready.", flush=True)

pool = ReplicaPool(_replicas, timeout=QUEUE_TIMEOUT)
print(f"[INIT] All {NUM_REPLICAS} replicas loaded. Queue timeout={QUEUE_TIMEOUT}s", flush=True)


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

@app.post("/v1/add_voice")
async def add_voice(body: AddVoiceRequest):
    """
    Register a voice for cloning across all replicas.
    The .wav file needs to already exist in /app (mounted to the root directory).
    """
    try:
        filepath = f"/app/{body.ref_audio_filename}"
        for replica in pool.replicas:
            prompt = replica.model.create_voice_clone_prompt(filepath, body.ref_text)
            replica.voice_clone_cache[filepath] = prompt
        return {"status": "success", "message": f"Voice added: {body.ref_audio_filename}"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech")
async def speech_endpoint(request: Request, body: SpeechRequest):
    text = body.input
    language = "English"

    replica = await pool.acquire()
    print(
        f"[REQ] /v1/audio/speech replica={replica.index} input: {text[:60]}",
        flush=True,
    )

    voice_clone_prompt = replica.voice_clone_cache[DEFAULT_VOICE_CLONE_REF_PATH]
    if body.cloning_audio_filename:
        print(f"[INFO] Using cloning audio: {body.cloning_audio_filename}", flush=True)
        voice_clone_prompt = replica.voice_clone_cache.get(
            body.cloning_audio_filename, voice_clone_prompt
        )

    def audio_stream():
        start = time.time()
        ttfb_printed = False
        try:
            for chunk, sr in replica.model.stream_generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                emit_every_frames=4,
                decode_window_frames=80,
                overlap_samples=512,
            ):
                if not ttfb_printed:
                    ttfb = time.time() - start
                    print(
                        f"[TTFB] replica={replica.index} {ttfb:.3f}s input: {text[:60]}",
                        flush=True,
                    )
                    ttfb_printed = True
                chunk_int16 = np.clip(chunk, -1.0, 1.0)
                chunk_int16 = (chunk_int16 * 32767.0).astype(np.int16).tobytes()
                yield chunk_int16
        except Exception as e:
            print(f"[ERROR] replica={replica.index} Exception: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            pool.release(replica)

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
        "replicas": NUM_REPLICAS,
        "queue_timeout": QUEUE_TIMEOUT,
    }
