"""Offline regression tests for the resemble-perth / pkg_resources fix.

Reproduces the production failure without torch: a miniature ``perth`` package
whose ``perth_net/__init__.py`` is byte-identical to resemble-perth 1.0.1's
first two lines. On any interpreter where ``pkg_resources`` is missing
(setuptools >= 82, and this VPS's bare python 3.11), the unpatched tree yields
``PerthImplicitWatermarker = None`` — the exact state that made chatterbox
raise ``TypeError: 'NoneType' object is not callable`` on RunPod.

No network, no torch, no GPU: safe for the CI ``test`` job.
"""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import patch_perth  # noqa: E402

UPSTREAM_PERTH_INIT = '''"""Perth: Audio Watermarking and Detection Library."""
from .watermarker import WatermarkerBase
from .dummy_watermarker import DummyWatermarker

try:
    from .perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
except ImportError:
    PerthImplicitWatermarker = None

__all__ = ['WatermarkerBase', 'DummyWatermarker']
'''

# Byte-identical to resemble-perth 1.0.1 (perth/perth_net/__init__.py).
UPSTREAM_PERTH_NET_INIT = (
    "from pkg_resources import resource_filename\n"
    'PREPACKAGED_MODELS_DIR = resource_filename(__name__, "pretrained")\n'
    "\n"
    "from .perth_net_implicit.perth_watermarker import PerthImplicitWatermarker\n"
)

STUB_WATERMARKER = "class PerthImplicitWatermarker:\n    pass\n"


def _build_tree(tmp_path: Path) -> Path:
    """A miniature resemble-perth install. Returns the directory to sys.path."""
    root = tmp_path / "fakepkg"
    perth = root / "perth"
    net = perth / "perth_net"
    implicit = net / "perth_net_implicit"
    pretrained = net / "pretrained" / "implicit"
    implicit.mkdir(parents=True)
    pretrained.mkdir(parents=True)

    (perth / "__init__.py").write_text(UPSTREAM_PERTH_INIT, encoding="utf-8")
    (perth / "watermarker.py").write_text("class WatermarkerBase:\n    pass\n", encoding="utf-8")
    (perth / "dummy_watermarker.py").write_text("class DummyWatermarker:\n    pass\n", encoding="utf-8")
    (net / "__init__.py").write_text(UPSTREAM_PERTH_NET_INIT, encoding="utf-8")
    (implicit / "__init__.py").write_text("", encoding="utf-8")
    (implicit / "perth_watermarker.py").write_text(STUB_WATERMARKER, encoding="utf-8")
    (pretrained / "perth_net_250000.pth.tar").write_bytes(b"fake-weights")
    return root


def _fresh_import(path_dir: Path):
    """Import ``perth`` from ``path_dir`` with a clean module cache."""
    for name in [n for n in sys.modules if n == "perth" or n.startswith("perth.")]:
        del sys.modules[name]
    sys.path.insert(0, str(path_dir))
    try:
        return importlib.import_module("perth")
    finally:
        sys.path.remove(str(path_dir))


def test_upstream_tree_reproduces_the_production_bug(tmp_path):
    """Without pkg_resources the upstream tree hides the failure as None."""
    if importlib.util.find_spec("pkg_resources") is not None:  # pragma: no cover
        pytest.skip("pkg_resources still available: upstream tree would import fine")
    root = _build_tree(tmp_path)
    perth = _fresh_import(root)
    assert perth.PerthImplicitWatermarker is None  # -> chatterbox: 'NoneType' not callable


def test_patch_restores_the_watermarker_and_the_model_dir(tmp_path):
    root = _build_tree(tmp_path)
    net_init = root / "perth" / "perth_net" / "__init__.py"

    assert patch_perth.patch_file(net_init) == "patched"
    perth = _fresh_import(root)

    assert perth.PerthImplicitWatermarker is not None
    assert Path(perth.perth_net.PREPACKAGED_MODELS_DIR) == (
        root / "perth" / "perth_net" / "pretrained"
    )
    assert (Path(perth.perth_net.PREPACKAGED_MODELS_DIR) / "implicit" / "perth_net_250000.pth.tar").is_file()


def test_patch_is_idempotent(tmp_path):
    root = _build_tree(tmp_path)
    net_init = root / "perth" / "perth_net" / "__init__.py"

    assert patch_perth.patch_file(net_init) == "patched"
    first = net_init.read_text(encoding="utf-8")
    assert patch_perth.patch_file(net_init) == "clean"
    assert net_init.read_text(encoding="utf-8") == first
    assert "from pkg_resources import" not in first


def test_leftover_pkg_resources_use_is_reported_not_ignored(tmp_path):
    root = _build_tree(tmp_path)
    stray = root / "perth" / "stray.py"
    stray.write_text("import pkg_resources\n", encoding="utf-8")

    assert patch_perth.patch_file(stray) == "leftover"
    assert patch_perth.main(["--root", str(root)]) == 2


def test_main_patches_an_installed_tree_and_is_quiet_when_clean(tmp_path, capsys):
    root = _build_tree(tmp_path)
    assert patch_perth.main(["--root", str(root)]) == 0
    assert "patched" in capsys.readouterr().out
    assert patch_perth.main(["--root", str(root)]) == 0
