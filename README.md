# tts-serverless-worker

RunPod Serverless worker for **Chatterbox Multilingual TTS** (Resemble AI, MIT) —
23 languages including Russian (`language_id: "ru"`), zero-shot voice cloning from
a short reference clip.

Built for kanban task **t_0e1043a2** (morning-brief voice over: compare against
`edge-tts ru-RU-SvetlanaNeural` and ElevenLabs). Image:
`ghcr.io/kirushanetypi/tts-serverless-worker:latest`.

The image is built by GitHub Actions (`.github/workflows/build.yml`) — the VPS has
~2 GB of free disk and cannot build a CUDA image locally.

## Why this model

| model | Russian | voice cloning | license | verdict |
|---|---|---|---|---|
| edge-tts `ru-RU-SvetlanaNeural` | native | no | Microsoft ToS, personal | current default, free, instant |
| F5-TTS russian finetune | yes (worse than edge) | yes | CC-BY-NC-4.0 | tested in round 3, lost on numbers |
| **Chatterbox Multilingual** | `ru` is an official language id | yes | **MIT** | this worker |

## Request

`POST https://api.runpod.ai/v2/<endpoint-id>/runsync` with

```json
{"input": {
  "text": "Доброе утро, Кирилл.",
  "language_id": "ru",
  "reference_audio": "<base64 wav/mp3 — optional, clones that speaker>",
  "t3_model": "v3",
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "temperature": 0.8,
  "repetition_penalty": 1.2,
  "min_p": 0.05,
  "top_p": 1.0,
  "seed": 20260911,
  "format": "mp3",
  "chunk_max_chars": 240,
  "pause_seconds": 0.22
}}
```

* `format` — `mp3` (default, 128 kbps, ~10x smaller JSON) or `wav`.
* `reference_audio` — base64 (or `data:audio/wav;base64,...`). ~10 s of clean
  speech is enough; the model truncates the prompt to `DEC_COND_LEN` (10 s at
  24 kHz) anyway. Without it the repo's built-in speaker (`conds.pt`) is used.
* `seed` — seeds torch so two runs of the same input are comparable.
* Long text is split on sentence boundaries into chunks of at most
  `chunk_max_chars`; the decoder emits at most 1000 speech tokens (~40 s of
  audio) per `generate` call, so an unsplit 35 s brief would be at risk of
  silent truncation.

## Response

```json
{"audio_base64": "...", "format": "mp3", "sample_rate": 24000,
 "meta": {"model_source": "runpod-host-cache", "model_load_seconds": 4.1,
          "generation_seconds": 8.3, "audio_seconds": 35.2,
          "generation_rtf": 0.24, "gpu_name": "NVIDIA RTX A4500",
          "chunks": 6, "per_chunk": [...], "request_seconds": 12.6}}
```

`meta` is the measurement surface: model provenance (`runpod-host-cache` vs
`hf-download`), cold-start load time, generation time, RTF, chunk count and the
GPU actually allocated. Nothing has to be inferred from logs.

## Cold start and cost

* `meta.model_source`:
  * `runpod-host-cache` — RunPod's *cached models* feature preloaded the repo on
    this host: the worker is not billed while the download happens and the load
    is seconds. Enable it per-endpoint in the console: **Manage → Edit Endpoint →
    Model → `ResembleAI/chatterbox`** (it is not a field on the REST API).
  * `local-cache` — found under the container's `HF_HOME`.
  * `hf-download` — downloaded now; `model_download_seconds` reports how long.
* Weights are ~3.1 GB for one T3 variant (`t3_mtl23ls_v3` 2.0 GB + `s3gen.pt`
  1.0 GB + `ve.pt`/tokenizer/conds), so the first cold start on an
  uncached host is dominated by that download, not by GPU work.

## Environment variables (endpoint template)

| var | default | meaning |
|---|---|---|
| `DEFAULT_LANGUAGE` | `ru` | `language_id` used when a job omits it |
| `CHATTERBOX_T3_MODEL` | `v3` | `v2`/`v3`/`t3_mtl23ls_v3`/explicit `.safetensors` |
| `CHATTERBOX_REPO` | `ResembleAI/chatterbox` | HF repo the checkpoints come from |
| `ENABLE_WATERMARK` | `1` | Resemble's Perth watermark on the audio |
| `ALLOW_DOWNLOAD` | `1` | `0` = fail instead of downloading weights |
| `DEFAULT_FORMAT` | `mp3` | `mp3` or `wav` |
| `FALLBACK_HF_HOME` | `/tmp/hf-home` | used when no host cache is mounted |

## Pitfalls found in production

* **PyPI `chatterbox-tts` 0.1.7 cannot select a T3 checkpoint.** Its
  `ChatterboxMultilingualTTS.from_local(ckpt_dir, device)` takes no `t3_model`
  argument and always loads `t3_mtl23ls_v2.safetensors` (git master accepts
  `t3_model`). The first real job on the deployed endpoint died with
  `TypeError: from_local() got an unexpected keyword argument 't3_model'`.
  `tts_utils.effective_t3_model()` now resolves the filename **the installed
  build will actually open** and the download follows it, so
  `CHATTERBOX_T3_MODEL=v3` degrades to v2 instead of fetching 2 GB the library
  will not load.
* **The decoder emits at most 1000 speech tokens per `generate` call** (~40 s of
  audio at 25 tokens/s), so long text is split on sentences; without that the
  tail of a brief is dropped silently.
* **A cloning request overwrites the model's speaker.** `model.conds` is
  restored from the repo's `conds.pt` before every job that passes no
  `reference_audio`, otherwise a later request inherits the previous caller's
  voice.
* **`resemble-perth` 1.0.1 died on setuptools >= 82.** It does
  `from pkg_resources import resource_filename` in
  `perth/perth_net/__init__.py`, and setuptools 82 removed `pkg_resources`
  (pypa/setuptools#5174). Its own `perth/__init__.py` catches the `ImportError`
  and silently sets `PerthImplicitWatermarker = None`, so chatterbox failed at
  `ChatterboxMultilingualTTS.__init__` with the opaque
  `TypeError: 'NoneType' object is not callable` — *after* downloading and
  loading the 3 GB checkpoints (RunPod job `2b6fb1ee`, delayTime 211.7 s).
  `patch_perth.py` rewrites that import into a stdlib `importlib.util` shim and
  the Dockerfile runs it with `--verify` right after the install, asserting both
  the class and its bundled 36 MB checkpoint load. A `setuptools<82` pin stays in
  `requirements.txt` as a second line of defence. Any *other* `pkg_resources`
  reference inside the package makes `patch_perth.py` exit 2, so a half-patched
  image fails the build instead of the job.

## Tests

```bash
python -m pytest tests -q     # 26 offline tests: chunking, base64/ref parsing, wav/mp3 I/O, perth patch
```

CI runs them in a separate job before the image build.
