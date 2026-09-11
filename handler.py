"""RunPod Serverless handler: Chatterbox Multilingual TTS (23 languages, ru included).

Why this worker exists (kanban t_0e1043a2): the morning brief is voiced by
``edge-tts ru-RU-SvetlanaNeural`` (fast, free, 7-8.5 % WER). Rounds 1-3 tested
piper/gTTS/F5-TTS as upgrades; F5 lost on Russian quality *and* on numbers
(digits come out as garbage) while costing 4-5.6 s of compute per second of
audio on the Mac. Chatterbox Multilingual (Resemble AI, MIT) is the strongest
open model with an official Russian language id, so this worker exists to test
it on a cheap RunPod GPU with the endpoint scaled to zero.

Request (``input``)::

    {"text": "...", "language_id": "ru",
     "reference_audio": "<base64 wav/mp3>",     # optional: clone this speaker
     "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8,
     "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0,
     "seed": 1234, "format": "mp3", "chunk_max_chars": 240}

Response::

    {"audio_base64": "...", "format": "mp3", "sample_rate": 24000, "meta": {...}}

``meta`` carries the numbers the report needs: model provenance + how long the
cold start took, generation seconds, RTF, chunk count and the GPU actually used.
"""
import base64
import hashlib
import logging
import os
import tempfile
import time

import numpy as np
import runpod
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from model_store import ensure_model, resolve_hf_home
from tts_utils import (
    MAX_TEXT_CHARS,
    SUPPORTED_LANGUAGES,
    clamp_float,
    clamp_int,
    effective_t3_model,
    encode_mp3,
    parse_reference,
    probe_duration,
    reference_suffix,
    split_text,
    t3_model_kwarg_supported,
    write_wav,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chatterbox-worker")

# ---- configuration (endpoint env vars) ------------------------------------ #
DEFAULT_LANGUAGE = (os.environ.get("DEFAULT_LANGUAGE") or "ru").strip().lower()
DEFAULT_T3_MODEL = (os.environ.get("CHATTERBOX_T3_MODEL") or "v3").strip()
REPO_ID = os.environ.get("CHATTERBOX_REPO", "ResembleAI/chatterbox")
ENABLE_WATERMARK = os.environ.get("ENABLE_WATERMARK", "1") not in ("0", "false", "False")
ALLOW_MODEL_DOWNLOAD = os.environ.get("ALLOW_DOWNLOAD", "1") not in ("0", "false", "False")
DEFAULT_FORMAT = (os.environ.get("DEFAULT_FORMAT") or "mp3").strip().lower()
MODEL_LOAD_TIMEOUT_NOTE = "first job on a cold worker pays the model load"

#: t3_model -> {"model", "load_seconds", "builtin_conds", "source", ...}
_MODELS = {}
#: reference-payload hash -> cached Conditionals for that speaker
_REFERENCE_CONDS = {}
#: per-process counters surfaced in meta
_STATS = {"jobs": 0}


def _now():
    return time.time()


def _gpu_name():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - telemetry must not fail a job
        return "cuda"


def _vram_gb():
    if not torch.cuda.is_available():
        return None, None
    try:
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return round(allocated, 2), round(total, 1)
    except Exception:  # noqa: BLE001
        return None, None


def load_model(t3_model):
    """Load (once per worker) the multilingual Chatterbox model."""
    if t3_model in _MODELS:
        entry = dict(_MODELS[t3_model])
        entry["reused"] = True
        return entry

    load_t3 = effective_t3_model(t3_model, ChatterboxMultilingualTTS.from_local)
    info = ensure_model(load_t3, repo_id=REPO_ID, allow_download=ALLOW_MODEL_DOWNLOAD)
    if info.get("error"):
        raise RuntimeError("checkpoint %s: %s" % (load_t3, info["error"]))

    log.info("loading model from %s (cached=%s download=%ss, requested t3=%s -> %s)",
             info["path"], info["cached"], info["download_seconds"], t3_model, load_t3)
    started = _now()
    if t3_model_kwarg_supported(ChatterboxMultilingualTTS.from_local):
        model = ChatterboxMultilingualTTS.from_local(info["path"], device="cuda",
                                                     t3_model=load_t3)
    else:
        model = ChatterboxMultilingualTTS.from_local(info["path"], device="cuda")
    load_seconds = round(_now() - started, 2)

    if not ENABLE_WATERMARK:
        model.watermarker = None
        log.warning("watermarker disabled by ENABLE_WATERMARK=0")

    _MODELS[t3_model] = {
        "model": model,
        # The default speaker that ships with the repo (conds.pt). A cloning
        # request overwrites model.conds, so the built-in one is kept here to
        # restore it for the next job that does not pass a reference clip.
        "builtin_conds": model.conds,
        "load_seconds": load_seconds,
        "source": info["source"],
        "cached": info["cached"],
        "download_seconds": info["download_seconds"],
        "bytes": info["bytes"],
        "path": info["path"],
        "hf_home": info["hf_home"],
        "reused": False,
    }
    log.info("model ready: load=%ss source=%s", load_seconds, info["source"])
    return dict(_MODELS[t3_model])


def _reference_path(payload, mime):
    suffix = reference_suffix(mime, payload)
    handle = tempfile.NamedTemporaryFile(prefix="tts-ref-", suffix=suffix, delete=False)
    try:
        handle.write(payload)
        return handle.name
    finally:
        handle.close()


def apply_reference(model, ref_path, exaggeration):
    """Use ``ref_path`` as the speaker, caching the computed conditionals.

    ``prepare_conditionals`` is not free (speaker embedding + speech tokens), so
    the same reference clip is only processed once per worker.
    """
    key = None
    if ref_path is not None:
        with open(ref_path, "rb") as handle:
            key = hashlib.sha1(handle.read()).hexdigest()
    if key is not None and key in _REFERENCE_CONDS:
        model.conds = _REFERENCE_CONDS[key]
        return {"reference_used": True, "reference_cached": True, "reference_sha1": key[:12]}
    model.prepare_conditionals(ref_path, exaggeration=exaggeration)
    if key is not None:
        _REFERENCE_CONDS[key] = model.conds
    return {"reference_used": True, "reference_cached": False,
            "reference_sha1": (key or "")[:12]}


def synthesize(model, chunks, language_id, params, pause_seconds=0.22):
    """Generate every chunk and stitch them into one float32 waveform."""
    sample_rate = model.sr
    pause = np.zeros(int(sample_rate * max(0.0, pause_seconds)), dtype=np.float32)
    pieces = []
    per_chunk = []
    for index, chunk in enumerate(chunks):
        started = _now()
        wav = model.generate(
            chunk,
            language_id,
            exaggeration=params["exaggeration"],
            cfg_weight=params["cfg_weight"],
            temperature=params["temperature"],
            repetition_penalty=params["repetition_penalty"],
            min_p=params["min_p"],
            top_p=params["top_p"],
        )
        elapsed = round(_now() - started, 2)
        audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        per_chunk.append({"chars": len(chunk), "seconds": elapsed,
                          "audio_seconds": round(len(audio) / sample_rate, 2)})
        if pieces:
            pieces.append(pause)
        pieces.append(audio)
    if not pieces:
        raise ValueError("nothing to synthesize")
    return np.concatenate(pieces), sample_rate, per_chunk


def handler(job):
    event = job or {}
    data = event.get("input") or {}
    if not isinstance(data, dict):
        raise ValueError("input must be an object")

    text = data.get("text") or data.get("prompt")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input.text must be a non-empty string")
    text = " ".join(text.split())
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("input.text exceeds %d characters" % MAX_TEXT_CHARS)

    language_id = str(data.get("language_id") or data.get("language") or DEFAULT_LANGUAGE).lower()
    if language_id not in SUPPORTED_LANGUAGES:
        raise ValueError("unsupported language_id %r; supported: %s"
                         % (language_id, ", ".join(sorted(SUPPORTED_LANGUAGES))))

    t3_model = str(data.get("t3_model") or DEFAULT_T3_MODEL)
    chunk_max_chars = clamp_int(data.get("chunk_max_chars"), 240, 60, 900)
    pause_seconds = clamp_float(data.get("pause_seconds"), 0.22, 0.0, 1.5)
    fmt = str(data.get("format") or DEFAULT_FORMAT).lower()
    if fmt not in ("mp3", "wav"):
        raise ValueError("format must be 'mp3' or 'wav'")

    params = {
        "exaggeration": clamp_float(data.get("exaggeration"), 0.5, 0.0, 2.0),
        "cfg_weight": clamp_float(data.get("cfg_weight"), 0.5, 0.0, 2.0),
        "temperature": clamp_float(data.get("temperature"), 0.8, 0.05, 2.0),
        "repetition_penalty": clamp_float(data.get("repetition_penalty"), 1.2, 1.0, 3.0),
        "min_p": clamp_float(data.get("min_p"), 0.05, 0.0, 1.0),
        "top_p": clamp_float(data.get("top_p"), 1.0, 0.05, 1.0),
    }
    seed = clamp_int(data.get("seed"), None, 0, 2 ** 31 - 1) if data.get("seed") is not None else None

    ref_bytes, ref_mime = parse_reference(data.get("reference_audio"))
    if ref_bytes is not None:
        # Reference clips come from callers (LibriVox public-domain readers,
        # existing TTS voices); only the payload is stored, in a temp dir.
        ref_path = _reference_path(ref_bytes, ref_mime)
    else:
        ref_path = None

    chunks = split_text(text, chunk_max_chars)
    request_started = _now()

    runpod.serverless.progress_update(event, "loading model %s" % t3_model)
    entry = load_model(t3_model)
    model = entry["model"]

    reference_meta = {"reference_used": False}
    if ref_path is not None:
        reference_meta = apply_reference(model, ref_path, params["exaggeration"])
        reference_meta["reference_bytes"] = len(ref_bytes)
    else:
        # Restore the repo's built-in speaker: a previous job may have left the
        # cloned speaker installed on this warm worker. ``generate`` adjusts
        # ``conds.t3.emotion_adv`` itself when ``exaggeration`` differs.
        model.conds = entry["builtin_conds"]

    if seed is not None:
        torch.manual_seed(seed)

    runpod.serverless.progress_update(event, "synthesizing %d chunk(s)" % len(chunks))
    generation_started = _now()
    audio, sample_rate, per_chunk = synthesize(
        model, chunks, language_id, params, pause_seconds=pause_seconds)
    generation_seconds = round(_now() - generation_started, 2)

    workdir = tempfile.mkdtemp(prefix="tts-out-")
    wav_path = os.path.join(workdir, "out.wav")
    write_wav(wav_path, audio, sample_rate)
    audio_seconds = probe_duration(wav_path) or round(len(audio) / sample_rate, 3)

    payload = None
    payload_format = fmt
    if fmt == "mp3":
        payload = encode_mp3(wav_path)
        if payload is None:
            log.warning("ffmpeg mp3 encode failed; returning WAV instead")
            payload_format = "wav"
    if payload is None:
        with open(wav_path, "rb") as handle:
            payload = handle.read()

    allocated_gb, total_gb = _vram_gb()
    _STATS["jobs"] += 1
    meta = {
        "language_id": language_id,
        "t3_model": t3_model,
        "model_repo": REPO_ID,
        "model_source": entry["source"],
        "model_cached": entry["cached"],
        "model_download_seconds": entry["download_seconds"],
        "model_load_seconds": entry["load_seconds"],
        "model_reused": entry["reused"],
        "model_bytes": entry["bytes"],
        "model_path": entry["path"],
        "hf_home": entry["hf_home"],
        "watermark": ENABLE_WATERMARK,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": _gpu_name(),
        "gpu_total_memory_gb": total_gb,
        "gpu_memory_allocated_gb": allocated_gb,
        "chunks": len(chunks),
        "per_chunk": per_chunk,
        "chunk_max_chars": chunk_max_chars,
        "pause_seconds": pause_seconds,
        "text_chars": len(text),
        "audio_seconds": audio_seconds,
        "generation_seconds": generation_seconds,
        "generation_rtf": round(generation_seconds / audio_seconds, 3) if audio_seconds else None,
        "request_seconds": round(_now() - request_started, 2),
        "format": payload_format,
        "audio_bytes": len(payload),
        "jobs_on_this_worker": _STATS["jobs"],
        "seed": seed,
        "params": params,
        "note": MODEL_LOAD_TIMEOUT_NOTE,
    }
    meta.update(reference_meta)
    log.info("done: %.2fs audio in %.2fs (rtf %.2f) lang=%s chunks=%d src=%s",
             audio_seconds, generation_seconds, meta["generation_rtf"] or -1,
             language_id, len(chunks), entry["source"])
    return {
        "audio_base64": base64.b64encode(payload).decode("ascii"),
        "format": payload_format,
        "sample_rate": sample_rate,
        "meta": meta,
    }


def _log_preflight():
    log.info("worker start: cuda=%s device=%s hf_home=%s default_language=%s t3=%s",
             torch.cuda.is_available(), _gpu_name(), resolve_hf_home(),
             DEFAULT_LANGUAGE, DEFAULT_T3_MODEL)


_log_preflight()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
