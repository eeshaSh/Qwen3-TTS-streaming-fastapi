"""
Lightweight FastAPI server exposing an OpenAI-compatible /v1/audio/speech endpoint for streaming TTS audio.

- Accepts POST requests with JSON: {"input": "text to synthesize"}
- Streams PCM audio bytes as response (audio/wav)
- Loads Qwen3TTSModel at startup for efficiency
"""

import io
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

app = FastAPI()

# Load model at startup (adjust path and params as needed)

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
# Enable streaming optimizations for maximum speed (matches test_streaming_optimized.py)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    use_cuda_graphs=False,  # Not needed with reduce-overhead mode
    compile_mode="reduce-overhead",
)

# Example voice clone prompt (replace with your own logic as needed)
ref_audio_path = "eesha_voice_cloning.wav"
ref_text = "Hello. This is an audio recording that's at least 5 seconds long. How are you doing today? Bye!"
voice_clone_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio_path, ref_text=ref_text)

class SpeechRequest(BaseModel):
    input: str
    # Optionally add model, voice, etc.

import time

@app.post("/v1/audio/speech")
async def speech_endpoint(request: Request, body: SpeechRequest):
    text = body.input
    language = "English"  # Or parse from request
    print(f"[REQ] /v1/audio/speech called with input: {text[:60]}", flush=True)

    def audio_stream():
        start = time.time()
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
                if not ttfb_printed:
                    ttfb = time.time() - start
                    print(f"[TTFB] /v1/audio/speech: {ttfb:.3f} seconds for input: {text[:60]}", flush=True)
                    ttfb_printed = True
                # Convert float32 [-1,1] to int16 PCM
                chunk_int16 = np.clip(chunk, -1.0, 1.0)
                chunk_int16 = (chunk_int16 * 32767.0).astype(np.int16)
                yield chunk_int16.tobytes()
        except Exception as e:
            print(f"[ERROR] Exception in audio_stream: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Optionally, yield nothing or a short silence to avoid abrupt close
            # silence = (np.zeros(2400, dtype=np.int16)).tobytes()  # 0.1s silence
            # yield silence
        finally:
            # Ensure generator ends cleanly
            pass

    # Set Content-Type to audio/L16 (PCM 16-bit signed)
    headers = {"Content-Type": "audio/L16; rate=24000"}
    return StreamingResponse(audio_stream(), headers=headers, media_type="audio/L16")

@app.get("/")
def root():
    return {"message": "Qwen3-TTS Streaming FastAPI server. POST /v1/audio/speech"}
