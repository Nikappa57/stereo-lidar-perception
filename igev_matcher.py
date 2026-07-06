"""Optional IGEV-Stereo learned matcher for the stereo-depth pipeline.

Registers an **IGEV-Stereo** (Xu et al., *Iterative Geometry Encoding Volume for
Stereo Matching*, CVPR 2023) matcher into :data:`data.LEARNED_MATCHERS`, so
``StereoSGBMConfig(matcher="igev")`` produces dense metric depth through the very
same :func:`data.stereo_depth` path as SGBM (rectify → match → depth → reproject).

The model code + weights are an **external, optional** dependency — exactly like
``ultralytics`` for the YOLO backbone. Nothing in the core preprocessing imports
this file; you opt in by calling :func:`register` (typically from a notebook or
the training script) before building the stereo branch.

Why IGEV here: KITTI-15 finetuned weights are domain-matched to KITTI-360.
Benchmarked vs SGBM on KITTI-360 (LiDAR GT): valid density **59% → ~100%** with
equal-or-better accuracy on the confident pixels (near/mid range wins; far range
is baseline-limited for both). See docs / the ``wls-depth-benchmark`` note.

Setup (once)
------------
1. Clone the official model code::

       git clone --depth 1 https://github.com/gangweiX/IGEV

2. Download the KITTI-15 finetuned weights, e.g. the HF mirror
   ``huggingface.co/shriarul5273/IGEV-Stereo`` → ``kitti/kitti15.pth``.
3. Install the feature-extractor dependency::

       pip install timm

4. Point this module at them (env vars, or pass paths to :func:`register`)::

       export IGEV_ROOT=/path/to/IGEV/IGEV-Stereo
       export IGEV_CKPT=/path/to/kitti15.pth

Usage
-----
::

    import data, igev_matcher
    igev_matcher.register()          # adds "igev" to data.LEARNED_MATCHERS
    sd = data.stereo_depth(sample, data.StereoSGBMConfig(matcher="igev"))

Notes
-----
* Input is the RGB rectified pair in ``[0, 255]``; IGEV normalises internally.
* ~0.5 s/frame at 1408×376, ``iters=16`` (32 gives no measurable gain here),
  <1 GB VRAM. Use :func:`data.precompute_stereo_inputs` to pay it once.
* Weights are loaded with ``weights_only=True`` (tensor-only; no pickle code).
* A recent ``timm`` fuses ``bn1+act1`` into ``BatchNormAct2d`` and drops
  ``act1``; :func:`_ensure_timm_compat` applies a one-line, idempotent patch to
  the clone's ``core/extractor.py`` so the pretrained backbone still loads.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

import data

_MODEL = None
_DEVICE = None
_PADDER = None  # IGEV's InputPadder class, loaded by file to dodge name clashes
_ROOT: str | None = None  # set by register(); falls back to $IGEV_ROOT / repo default
_CKPT: str | None = None  # set by register(); falls back to $IGEV_CKPT / repo default

# Repo-local defaults: the IGEV model code (its ``core/``) and KITTI-15 weights
# vendored under ``third_party/`` (gitignored). Resolution order is:
# explicit arg > env var > this repo-local default. So the notebooks / training
# work out of the box once ``third_party/`` is populated, with no env vars.
_REPO = Path(__file__).resolve().parent
_DEFAULT_ROOT = _REPO / "third_party" / "IGEV-Stereo"
_DEFAULT_CKPT = _REPO / "third_party" / "igev_kitti15.pth"


def _resolve_paths(igev_root: str | None, ckpt: str | None) -> tuple[Path, Path]:
    root = Path(igev_root or os.environ.get("IGEV_ROOT") or _DEFAULT_ROOT).expanduser()
    weights = Path(ckpt or os.environ.get("IGEV_CKPT") or _DEFAULT_CKPT).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"IGEV model code not found at {root}. Vendor the IGEV-Stereo "
            "'core/' into third_party/IGEV-Stereo/, or set IGEV_ROOT "
            "(see module docstring).")
    if not weights.exists():
        raise FileNotFoundError(
            f"IGEV weights not found at {weights}. Put kitti15.pth at "
            "third_party/igev_kitti15.pth, or set IGEV_CKPT.")
    return root, weights


def _ensure_timm_compat(igev_root: Path) -> None:
    """Idempotently make ``core/extractor.py`` tolerate a modern ``timm``.

    Newer ``timm`` folds the stem activation into ``bn1`` (``BatchNormAct2d``)
    and no longer exposes ``model.act1``, which crashes the stock IGEV
    ``Feature`` extractor. Replace the hard attribute access with a fallback.
    """
    ext = igev_root / "core" / "extractor.py"
    src = ext.read_text()
    old = "        self.act1 = model.act1\n"
    new = ("        self.act1 = model.act1 if hasattr(model, 'act1') "
           "else torch.nn.Identity()\n")
    if old in src and new not in src:
        ext.write_text(src.replace(old, new))


def get_model(igev_root: str | None = None, ckpt: str | None = None,
              device: str | None = None):
    """Lazily build IGEV-Stereo and load the (tensor-only) checkpoint."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL

    import argparse
    root, weights = _resolve_paths(igev_root or _ROOT, ckpt or _CKPT)
    _ensure_timm_compat(root)
    for p in (str(root), str(root / "core")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from igev_stereo import IGEVStereo  # type: ignore  # noqa: E402  (vendored in third_party/IGEV-Stereo, added to sys.path just above)

    _DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.Namespace(
        hidden_dims=[128] * 3, corr_levels=2, corr_radius=4,
        n_downsample=2, n_gru_layers=3, max_disp=192,
        mixed_precision=(_DEVICE == "cuda"), precision_dtype="float16",
    )
    model = IGEVStereo(args)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    _MODEL = model.to(_DEVICE).eval()
    return _MODEL


def _input_padder():
    """Load IGEV's ``InputPadder`` by file path.

    The IGEV clone ships a ``core/utils/utils.py`` whose top-level ``utils`` name
    collides with this project's own ``utils.py`` (which the notebooks import).
    Loading the class straight from its file under a private module name avoids
    the clash regardless of import order or what is already in ``sys.modules``.
    """
    global _PADDER
    if _PADDER is None:
        import importlib.util
        root, _ = _resolve_paths(_ROOT, _CKPT)
        path = root / "core" / "utils" / "utils.py"
        spec = importlib.util.spec_from_file_location("_igev_padder_utils", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PADDER = mod.InputPadder
    return _PADDER


@torch.no_grad()
def igev_disparity(rect_left: np.ndarray, rect_right: np.ndarray,
                   cfg=None, iters: int | None = None) -> np.ndarray:
    """Rectified RGB pair ``(H, W, 3)`` uint8 → left disparity ``(H, W)`` px.

    Matches the ``data.LEARNED_MATCHERS`` contract ``(rect_left, rect_right,
    cfg) -> disparity``. IGEV is fully dense, so every pixel gets a positive
    disparity; the metric-range / BEV filters downstream drop sky/far outliers.
    """
    model = get_model()  # also sets up sys.path + resolves _ROOT/_CKPT
    InputPadder = _input_padder()
    n_iters = iters if iters is not None else int(os.environ.get("IGEV_ITERS", 16))

    def to_t(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        return t[None].to(_DEVICE)

    im1, im2 = to_t(rect_left), to_t(rect_right)
    padder = InputPadder(im1.shape, divis_by=32)
    im1, im2 = padder.pad(im1, im2)
    disp = model(im1, im2, iters=n_iters, test_mode=True)  # (1,1,H,W) positive px
    disp = padder.unpad(disp).squeeze().float().cpu().numpy()
    return disp.astype(np.float32)


def register(name: str = "igev", igev_root: str | None = None,
             ckpt: str | None = None) -> None:
    """Register the IGEV matcher under ``name`` in :data:`data.LEARNED_MATCHERS`.

    Paths resolve from the args or the ``IGEV_ROOT`` / ``IGEV_CKPT`` env vars.
    The model builds lazily on first use, so this call is cheap; passing explicit
    paths validates them now (fail fast) and remembers them for the lazy build.
    """
    global _ROOT, _CKPT
    if igev_root is not None:
        _ROOT = igev_root
    if ckpt is not None:
        _CKPT = ckpt
    if igev_root or ckpt:  # fail fast on obviously wrong paths
        _resolve_paths(igev_root or _ROOT, ckpt or _CKPT)
    data.LEARNED_MATCHERS[name] = igev_disparity
