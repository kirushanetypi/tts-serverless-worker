"""Locate (and if needed download) the Chatterbox checkpoint files.

Two goals:

1. **Use the host's cached model when there is one.** RunPod's *cached models*
   feature preloads a Hugging Face repo onto the host and exposes it read-only
   at ``/runpod-volume/huggingface-cache/hub/models--<org>--<name>/snapshots/<hash>/``,
   and the worker is not billed while that download happens. The feature is
   enabled per-endpoint in the console (it is not a field on the REST API), so
   this module simply *detects* it at runtime: when the snapshot is already
   there, nothing is downloaded and the cold start is seconds instead of
   minutes.

2. **Report what happened.** ``ensure_model`` returns where the weights came
   from, how long it took and how big they were, so every job response carries
   the cold-start evidence instead of it having to be guessed from logs.
"""
import os
import time
from pathlib import Path

HOST_CACHE_ROOT = "/runpod-volume/huggingface-cache"
REPO_ID = os.environ.get("CHATTERBOX_REPO", "ResembleAI/chatterbox")
FALLBACK_HF_HOME = os.environ.get("FALLBACK_HF_HOME", "/tmp/hf-home")

#: Files the multilingual model needs. ``t3_model`` is added per request, so a
#: worker that switches between v2/v3 does not have to hold both in memory.
BASE_FILES = ("ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt")


def resolve_hf_home():
    """Pick ``HF_HOME`` and export it, so huggingface_hub uses the same path.

    Order: an HF_HOME RunPod already set > a mounted cached-model tree > the
    container-local fallback.
    """
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    host_hub = Path(HOST_CACHE_ROOT) / "hub"
    if host_hub.is_dir():
        os.environ["HF_HOME"] = HOST_CACHE_ROOT
        return HOST_CACHE_ROOT
    os.environ["HF_HOME"] = FALLBACK_HF_HOME
    return FALLBACK_HF_HOME


def repo_cache_dir(repo_id=None, hf_home=None):
    """HF cache directory huggingface_hub uses for ``repo_id``."""
    hf_home = hf_home or resolve_hf_home()
    name = "models--" + (repo_id or REPO_ID).replace("/", "--")
    return Path(hf_home) / "hub" / name


def snapshot_dirs(repo_id=None, hf_home=None):
    """Every ``snapshots/<hash>`` directory for the repo, newest first."""
    snaps = repo_cache_dir(repo_id, hf_home) / "snapshots"
    if not snaps.is_dir():
        return []
    dirs = [d for d in snaps.iterdir() if d.is_dir()]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def required_files(t3_model):
    return tuple(BASE_FILES) + (t3_model,)


def complete_snapshot(t3_model, repo_id=None, hf_home=None):
    """Newest snapshot that contains every file we need, or ``None``."""
    need = set(required_files(t3_model))
    for snap in snapshot_dirs(repo_id, hf_home):
        present = {p.name for p in snap.iterdir() if p.is_file()}
        if need.issubset(present):
            return snap
    return None


def dir_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def ensure_model(t3_model, repo_id=None, hf_home=None, allow_download=True, token=None):
    """Return the local snapshot directory for ``t3_model``.

    Never re-downloads: a complete on-host/on-disk snapshot short-circuits the
    call. The returned dict carries the provenance used in job ``meta``.
    """
    repo_id = repo_id or REPO_ID
    hf_home = hf_home or resolve_hf_home()
    started = time.time()

    snap = complete_snapshot(t3_model, repo_id, hf_home)
    if snap is not None:
        source = "runpod-host-cache" if str(hf_home).startswith(HOST_CACHE_ROOT) else "local-cache"
        return {
            "path": str(snap),
            "cached": True,
            "source": source,
            "download_seconds": 0.0,
            "seconds": round(time.time() - started, 2),
            "bytes": dir_bytes(snap),
            "hf_home": str(hf_home),
            "repo_id": repo_id,
            "t3_model": t3_model,
        }

    if not allow_download:
        return {"error": "no cached snapshot and ALLOW_DOWNLOAD=0",
                "hf_home": str(hf_home), "repo_id": repo_id, "t3_model": t3_model}

    from huggingface_hub import snapshot_download  # imported late: slow import

    dl_started = time.time()
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        allow_patterns=list(required_files(t3_model)),
        token=token or os.environ.get("HF_TOKEN"),
    )
    download_seconds = round(time.time() - dl_started, 2)
    return {
        "path": str(path),
        "cached": False,
        "source": "hf-download",
        "download_seconds": download_seconds,
        "seconds": round(time.time() - started, 2),
        "bytes": dir_bytes(path),
        "hf_home": str(hf_home),
        "repo_id": repo_id,
        "t3_model": t3_model,
    }
