"""Precompute the IGEV stereo-depth cache for both splits.

Idempotent: frames already in the cache are skipped, so this only builds the
newly added drive (0009). Run once, before training the camera / pipeline_*
models, so the training loop reads the cache with no matcher at runtime.

    python scripts/build_igev_cache.py            # both splits, IGEV
    MATCHER=sgbm python scripts/build_igev_cache.py
"""
import os
import sys
from pathlib import Path

# Repo root on sys.path + point py123d at the (symlinked) roots.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
os.environ.setdefault("PY123D_DATA_ROOT", str(_REPO / "data"))
os.environ.setdefault("KITTI360_DATA_ROOT", str(_REPO / "KITTI-360"))

from data import (Py123dDataset, StereoSGBMConfig,  # noqa: E402
                  precompute_stereo_inputs, stereo_cache_root)

MATCHER = os.environ.get("MATCHER", "igev")

if MATCHER != "sgbm":
    import igev_matcher  # noqa: E402
    igev_matcher.register()

train_ds = Py123dDataset(split_names=["kitti360_train"])
val_ds = Py123dDataset(split_names=["kitti360_val"])
cache_root = stereo_cache_root(train_ds.data_root, matcher=MATCHER)
print(f"matcher={MATCHER} | cache_root={cache_root}", flush=True)

for name, ds in (("train", train_ds), ("val", val_ds)):
    print(f"=== {name}: {len(ds)} frames / {ds.scene_count} drives ===",
          flush=True)
    precompute_stereo_inputs(ds, cache_root=cache_root,
                             sgbm_cfg=StereoSGBMConfig(matcher=MATCHER))

print("CACHE BUILD DONE", flush=True)
