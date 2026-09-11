"""Pure-python helpers for the Chatterbox TTS worker.

Deliberately free of torch/runpod imports so the whole module (and its tests)
runs on a laptop, in CI, and on the VPS without a GPU.
"""
import base64
import binascii
import inspect
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

#: Mirrors chatterbox.mtl_tts.SUPPORTED_LANGUAGES. Kept here so the request can
#: be rejected before the model is loaded (a job that fails after a cold start
#: wastes GPU time).
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

#: The decoder emits at most max_new_tokens=1000 speech tokens, one token per
#: 40 ms of audio (S3GEN_SR 24000 / S3_TOKEN_RATE 25), i.e. ~40 s of speech per
#: ``generate`` call. A longer text silently truncates, so text is split into
#: sentence-sized chunks that stay well inside that budget.
MAX_CHUNK_CHARS = 240

MAX_TEXT_CHARS = 4000
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")


def clamp_float(value, default, low, high):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if num != num:  # NaN
        return default
    return max(low, min(high, num))


def clamp_int(value, default, low, high):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def split_text(text, max_chars=MAX_CHUNK_CHARS):
    """Split ``text`` into chunks no longer than ``max_chars``.

    Sentence boundaries first; a single oversized sentence is then broken on
    commas/semicolons and, failing that, on whitespace. Nothing is ever dropped:
    ``"".join(split_text(t))``-style loss is prevented by keeping the original
    characters, only whitespace at the seams is trimmed.
    """
    text = " ".join((text or "").split())
    if not text:
        return []

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s]
    chunks = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sentence) <= max_chars:
            current = sentence
            continue
        chunks.extend(_hard_split(sentence, max_chars))
    if current:
        chunks.append(current)
    return chunks


def _hard_split(sentence, max_chars):
    """Break one oversized sentence on soft punctuation, then on words."""
    parts = re.split(r"(?<=[,;:—–])\s+", sentence)
    out = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            out.append(current)
            current = ""
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        current = part
    if current:
        out.append(current)
    return [c for c in out if c]


def t3_model_kwarg_supported(from_local):
    """Does this chatterbox build take a ``t3_model`` argument?

    PyPI ``chatterbox-tts`` 0.1.7 hardcodes ``t3_mtl23ls_v2.safetensors`` and its
    ``from_local(ckpt_dir, device)`` rejects a ``t3_model`` keyword — the first
    job on the deployed endpoint really died with
    ``TypeError: from_local() got an unexpected keyword argument 't3_model'``.
    Git master accepts it, so both shapes are supported.
    """
    try:
        return "t3_model" in inspect.signature(from_local).parameters
    except (TypeError, ValueError):  # builtins / C-level callables
        return False


#: Short names accepted by the model library -> checkpoint file in the HF repo.
T3_MODEL_FILES = {
    "v2": "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v2": "t3_mtl23ls_v2.safetensors",
    "v3": "t3_mtl23ls_v3.safetensors",
    "t3_mtl23ls_v3": "t3_mtl23ls_v3.safetensors",
}
DEFAULT_T3_FILE = "t3_mtl23ls_v2.safetensors"


def resolve_t3_model_file(requested):
    """Map ``v3``/``t3_mtl23ls_v3``/``foo.safetensors`` to a real repo filename."""
    if not requested:
        return DEFAULT_T3_FILE
    requested = str(requested).strip()
    if requested.endswith(".safetensors"):
        return requested
    return T3_MODEL_FILES.get(requested, DEFAULT_T3_FILE)


def effective_t3_model(requested, from_local):
    """The checkpoint **filename** that will actually be loaded.

    Two things have to agree: what gets downloaded and what the library loads.
    A build that cannot select a T3 variant always loads v2, so the download has
    to follow the library instead of the request — otherwise a 0.1.7 worker
    fetches v3 and then looks for a missing v2.
    """
    file_name = resolve_t3_model_file(requested)
    if from_local is None or t3_model_kwarg_supported(from_local):
        return file_name
    return DEFAULT_T3_FILE


def parse_reference(value):
    """Decode a reference clip supplied as base64 or a ``data:`` URI.

    Returns ``(bytes, mime_hint)``. Raises ``ValueError`` on anything that is
    not decodable audio payload.
    """
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reference_audio must be a non-empty base64 string")
    encoded = value.strip()
    mime = None
    if encoded.startswith("data:"):
        head, _, encoded = encoded.partition(",")
        mime = head[5:].split(";")[0] or None
        if not encoded:
            raise ValueError("reference_audio data URI carries no payload")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("reference_audio is not valid base64") from exc
    if not payload:
        raise ValueError("reference_audio decoded to 0 bytes")
    if len(payload) > MAX_REFERENCE_BYTES:
        raise ValueError("reference_audio exceeds 12 MB")
    return payload, mime


def reference_suffix(mime, payload):
    """File suffix for a decoded reference clip (ffmpeg sniffs regardless)."""
    known = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3",
             "audio/mp3": ".mp3", "audio/flac": ".flac", "audio/x-flac": ".flac",
             "audio/ogg": ".ogg", "audio/webm": ".webm", "audio/mp4": ".m4a"}
    if mime in known:
        return known[mime]
    for magic, suffix in ((b"RIFF", ".wav"), (b"ID3", ".mp3"), (b"fLaC", ".flac"),
                          (b"OggS", ".ogg")):
        if payload.startswith(magic):
            return suffix
    if payload[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    return ".audio"


def write_wav(path, samples, sample_rate):
    """Write mono int16 PCM from a float sequence in [-1, 1]."""
    clipped = bytearray()
    for value in samples:
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        clipped += int(value * 32767.0).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(bytes(clipped))
    return str(path)


def read_wav_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def encode_mp3(wav_path, bitrate="128k", timeout=180):
    """ffmpeg wav -> mp3. Returns bytes, or ``None`` when ffmpeg is absent/fails.

    MP3 keeps the JSON job response ~10x smaller than WAV (a 35 s brief is
    ~2.3 MB as base64 WAV but ~0.25 MB as 128 kbps MP3).
    """
    out_path = Path(tempfile.mkdtemp(prefix="tts-mp3-")) / (Path(wav_path).stem + ".mp3")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate,
           "-ac", "1", str(out_path)]
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return None
    if not out_path.is_file() or out_path.stat().st_size == 0:
        return None
    return out_path.read_bytes()


def probe_duration(path):
    """Audio duration in seconds via ffprobe, or ``None`` when unavailable."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "json", str(path)]
    try:
        done = subprocess.run(cmd, check=True, timeout=60, capture_output=True, text=True)
        return round(float(json.loads(done.stdout)["format"]["duration"]), 3)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        return None
