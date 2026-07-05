"""Project-wide configuration — the single source of truth.

Everything that has to agree across the two branches and the fusion lives here:
the shared **BEV grid** (extent + resolution), the **channel contract** between
Stage A and the fusion, and the **class set**. Every per-module config dataclass
(`PillarConfig`, `MonoBEVConfig`, `StereoBEVConfig`, `BEVFusionConfig`, …)
defaults to these constants instead of re-declaring the numbers, so the grid and
the 64/128 channel contract can never drift between modules (design doc §02; was
previously duplicated across four configs — see TODO P0).

The shared grid::

    x ∈ [0, 50] m (forward), y ∈ [-20, 20] m (lateral), z ∈ [-3, 1] m
    0.25 m / cell  ->  nx = 200, ny = 160      (indexing: flat = ix * ny + iy)

Note: this module is named ``globals.py`` to match the project layout. It does
**not** shadow the builtin ``globals()`` function — that is only reachable by
name, and ``import globals`` binds the module, not the builtin.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Shared BEV grid (must be identical across every branch and the fusion)
# --------------------------------------------------------------------------- #
X_RANGE: tuple[float, float] = (0.0, 50.0)  # forward (ego +x)
Y_RANGE: tuple[float, float] = (-20.0, 20.0)  # lateral (ego +y)
Z_RANGE: tuple[float, float] = (-3.0, 1.0)  # height pre-filter
BEV_RES_M: float = 0.25  # metres per BEV cell


def grid_size(
    x_range: tuple[float, float] = X_RANGE,
    y_range: tuple[float, float] = Y_RANGE,
    res_m: float = BEV_RES_M,
) -> tuple[int, int]:
    """`(nx, ny)` cell counts for a BEV extent + resolution. Defaults = shared grid."""
    nx = int(round((x_range[1] - x_range[0]) / res_m))
    ny = int(round((y_range[1] - y_range[0]) / res_m))
    return nx, ny


GRID_SIZE: tuple[int, int] = grid_size()  # (200, 160) with defaults

# --------------------------------------------------------------------------- #
# Channel contract between Stage A (the two branches) and the fusion + head
# --------------------------------------------------------------------------- #
CAMERA_BEV_CHANNELS: int = 64  # C_cam — StereoBEV/MonoBEV context width
LIDAR_BEV_CHANNELS: int = 128  # C_lidar — PointPillars BEVBackbone2D output
FUSED_CHANNELS: int = 128  # fused feature depth fed to the head

# --------------------------------------------------------------------------- #
# Dataset selection — single source of truth (flip these to swap datasets)
# --------------------------------------------------------------------------- #
# The pipeline is dataset-agnostic (everything goes through py123d's SceneAPI);
# only the split names + stereo camera ids change between datasets. KITTI-360 is
# the primary dataset because it is the only py123d source with a real *colour*
# stereo pair (AV2's stereo pair is grayscale). See docs/dataset.md.
DATASET: str = "kitti360"
TRAIN_SPLIT: str = "kitti360_train"
VAL_SPLIT: str = "kitti360_val"
# Stereo pair camera ids. KITTI-360 maps image_00/image_01 → these same enums,
# so the id names are shared with AV2; only the split names above differ.
LEFT_CAMERA: str = "pcam_stereo_l"
RIGHT_CAMERA: str = "pcam_stereo_r"

# --------------------------------------------------------------------------- #
# Classes (design doc §05)
# --------------------------------------------------------------------------- #
# The py123d loader emits a *unified* taxonomy (VEHICLE / PERSON / BARRIER /
# TWO_WHEELER / TRAFFIC_CONE / TRAFFIC_SIGN / OTHER / ANIMAL), coarser than each
# dataset's raw names. We train on a 3-class subset. KITTI-360 has **no cones**
# (urban German streets), so the small-object slot is TWO_WHEELER (bicycle /
# motorcycle / rider) — which py123d does *not* fold into VEHICLE. Note this
# means VEHICLE = car/truck/bus only. Everything else maps to the ignore bucket
# (index None). TRAFFIC_CONE remains the Formula-Student deployment target, but
# it must come from AV2 (CONSTRUCTION_CONE) or CARLA — swap it back in there.
CLASSES: tuple[str, ...] = ("VEHICLE", "PERSON", "TWO_WHEELER")
NUM_CLASSES: int = len(CLASSES)

# Loader label -> training class index. Labels absent here are ignored (not a
# negative, not a positive — dropped before encoding/loss).
CLASS_REMAP: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


def class_index(label: str) -> int | None:
    """Map a loader label to its training class index, or ``None`` to ignore it."""
    return CLASS_REMAP.get(label)
