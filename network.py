"""BEV fusion + 2D detection head — the consumer side of Pipeline A / C.

The design's core principle (doc §02): two feature maps may be fused cell-by-cell
only if their cells refer to the same physical place. The shared ego BEV grid is
that place, so this module takes the two **grid-aligned** BEV maps emitted by
Stage A and fuses them, then a CenterPoint-style head reads object centres off
the fused map.

Building this block first **fixes the contract for Stage A**: the camera branch
(grounded stereo-depth splat) and the LiDAR branch (PointPillars) must each emit
a ``(B, C, nx, ny)`` tensor on the shared grid. ``camera_channels`` /
``lidar_channels`` / ``grid_size`` below are exactly what those branches have to
output — use :meth:`BEVDetector.from_bev_maps` to build a detector straight from
the branch outputs, and :func:`describe` to print the contract + parameter count.

The fusion block has a **fixed interface** (two BEV maps in, one out), so it is
swapped per pipeline without touching the branches or the head (doc §07/§08):

  * A : :class:`ConcatConvFusion` (concat + conv).
  * B : :class:`ConcatConvFusion` too — B's BEV fusion is *identical* to A. B
        differs only upstream, in Stage A (a painted LiDAR-range channel added
        to the image fusion); the BEV fusion block does not change.
  * C : :class:`CrossAttentionFusion` (near->stereo, far->LiDAR) — drop-in swap.
  * D (late fusion) has **no** BEV fusion: it merges object lists, not feature
    maps, so it does not use this module at all.

Output is **2D only**: centre (x, y) + class. No yaw, no z (doc §08).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Type

import torch
import torch.nn as nn

# Stage A output channels — single source of truth. Keep in sync with:
#   camera: StereoBEVConfig.context_channels / MonoBEVConfig.context_channels
#   lidar : PointPillarsBranch -> BEVBackbone2D(out_channels=128)
CAMERA_BEV_CHANNELS = 64
LIDAR_BEV_CHANNELS = 128


def _as_batched(x: torch.Tensor) -> torch.Tensor:
    """Accept a branch output as ``(C, nx, ny)`` or ``(B, C, nx, ny)``."""
    return x.unsqueeze(0) if x.dim() == 3 else x


def num_parameters(module: nn.Module) -> int:
    """Total trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass
class BEVFusionConfig:
    """Contract between Stage A (the two branches) and the fusion + head.

    Defaults match the current branches and the shared grid
    (x in [0, 50] m, y in [-20, 20] m, 0.25 m/cell -> nx=200, ny=160).
    """

    camera_channels: int = CAMERA_BEV_CHANNELS  # C_cam Stage A must emit
    lidar_channels: int = LIDAR_BEV_CHANNELS    # C_lidar Stage A must emit
    out_channels: int = 128                     # fused feature depth -> head
    grid_size: Tuple[int, int] = (200, 160)     # (nx, ny), shared by both branches
    num_classes: int = 3                        # set by the class filter (doc §05)
    head_channels: int = 64

    @classmethod
    def from_bev_maps(
        cls, bev_camera: torch.Tensor, bev_lidar: torch.Tensor, **overrides
    ) -> "BEVFusionConfig":
        """Read the contract straight off the two Stage A BEV tensors.

        This is the "go back to Stage A" loop: whatever channels/grid the
        branches actually produce become the fusion's expected inputs.
        """
        cam, lid = _as_batched(bev_camera), _as_batched(bev_lidar)
        return cls(
            camera_channels=cam.shape[1],
            lidar_channels=lid.shape[1],
            grid_size=(int(cam.shape[-2]), int(cam.shape[-1])),
            **overrides,
        )


class BEVFusion(nn.Module):
    """Abstract fusion block: two grid-aligned BEV maps -> one fused map.

    Subclasses implement :meth:`_fuse`. The shape guard in :meth:`forward` makes
    doc §02 executable — fusion only runs when both maps share the grid and carry
    the channel counts the config promised, so a Stage A change can't silently
    feed misaligned maps.
    """

    def __init__(self, cfg: BEVFusionConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, bev_camera: torch.Tensor, bev_lidar: torch.Tensor) -> torch.Tensor:
        cam, lid = _as_batched(bev_camera), _as_batched(bev_lidar)
        assert cam.shape[0] == lid.shape[0], f"batch mismatch: {cam.shape[0]} vs {lid.shape[0]}"
        assert cam.shape[-2:] == lid.shape[-2:], (
            f"BEV grids not aligned: camera {tuple(cam.shape[-2:])} vs lidar "
            f"{tuple(lid.shape[-2:])} — cannot fuse cell-by-cell (doc §02)"
        )
        assert cam.shape[1] == self.cfg.camera_channels, (
            f"camera BEV has {cam.shape[1]} ch, fusion expects {self.cfg.camera_channels}"
        )
        assert lid.shape[1] == self.cfg.lidar_channels, (
            f"lidar BEV has {lid.shape[1]} ch, fusion expects {self.cfg.lidar_channels}"
        )
        return self._fuse(cam, lid)

    def _fuse(self, cam: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConcatConvFusion(BEVFusion):
    """Pipeline A fusion: channel-concatenate the two BEV maps, then convolve."""

    def __init__(self, cfg: BEVFusionConfig):
        super().__init__(cfg)
        c_in = cfg.camera_channels + cfg.lidar_channels
        self.block = nn.Sequential(
            nn.Conv2d(c_in, cfg.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(cfg.out_channels, cfg.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.out_channels), nn.ReLU(inplace=True),
        )

    def _fuse(self, cam: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([cam, lid], dim=1))  # (B, out_channels, nx, ny)


class CrossAttentionFusion(BEVFusion):
    """Pipeline C fusion — same interface, learns near->stereo / far->lidar.

    Drop-in replacement for :class:`ConcatConvFusion` (doc §07 C). Use
    deformable / windowed attention or attend only at candidate cells; full
    all-to-all over the grid is quadratic and too slow.
    """

    def _fuse(self, cam: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Pipeline C cross-attention fusion: TODO (P4).")


class CenterPointHead(nn.Module):
    """2D BEV CenterPoint head: fused map -> per-class centre heatmap + (x,y) offset.

    No yaw / z regression (doc §08). The heatmap conv outputs logits (apply a
    sigmoid in the focal loss / at inference); the offset is the sub-cell
    (dx, dy) of the centre within its grid cell.
    """

    def __init__(self, in_channels: int, num_classes: int, head_channels: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels), nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(head_channels, num_classes, 1)
        self.offset = nn.Conv2d(head_channels, 2, 1)
        # Focal-loss prior: start heatmap near p~=0.1 so training isn't swamped
        # by the overwhelmingly empty grid (CenterPoint convention).
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared(fused)
        return {"heatmap": self.heatmap(feat), "offset": self.offset(feat)}


class BEVDetector(nn.Module):
    """Pipeline A consumer: BEV fusion + CenterPoint head.

    ``forward(bev_camera, bev_lidar) ->
        {"heatmap": (B, num_classes, nx, ny), "offset": (B, 2, nx, ny)}``

    It deliberately does **not** own the Stage A branches — they are a separate,
    swappable concern (and the camera branch's stereo-depth front-end is still
    in progress). Feed it the two branch outputs.
    """

    def __init__(self, cfg: BEVFusionConfig = None, fusion_cls: Type[BEVFusion] = ConcatConvFusion):
        super().__init__()
        self.cfg = cfg or BEVFusionConfig()
        self.fusion = fusion_cls(self.cfg)
        self.head = CenterPointHead(self.cfg.out_channels, self.cfg.num_classes, self.cfg.head_channels)

    @classmethod
    def from_bev_maps(
        cls,
        bev_camera: torch.Tensor,
        bev_lidar: torch.Tensor,
        num_classes: int = 3,
        fusion_cls: Type[BEVFusion] = ConcatConvFusion,
        **overrides,
    ) -> "BEVDetector":
        """Build a detector whose fusion inputs match the given Stage A outputs."""
        cfg = BEVFusionConfig.from_bev_maps(bev_camera, bev_lidar, num_classes=num_classes, **overrides)
        return cls(cfg, fusion_cls=fusion_cls)

    def forward(self, bev_camera: torch.Tensor, bev_lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.head(self.fusion(bev_camera, bev_lidar))


def describe(detector: BEVDetector) -> None:
    """Print the Stage A contract and per-module parameter counts."""
    c = detector.cfg
    nx, ny = c.grid_size
    print("BEV fusion contract — Stage A must emit, on the shared grid:")
    print(f"  camera BEV : (B, {c.camera_channels}, {nx}, {ny})")
    print(f"  lidar  BEV : (B, {c.lidar_channels}, {nx}, {ny})")
    print(f"  fused      : (B, {c.out_channels}, {nx}, {ny})")
    print(f"  head out   : heatmap (B, {c.num_classes}, {nx}, {ny}) + offset (B, 2, {nx}, {ny})")
    print("parameters:")
    print(f"  fusion : {num_parameters(detector.fusion):,}")
    print(f"  head   : {num_parameters(detector.head):,}")
    print(f"  total  : {num_parameters(detector):,}")


if __name__ == "__main__":
    # Smoke test with random Stage A maps (no dataset needed).
    nx, ny = 200, 160
    bev_cam = torch.randn(CAMERA_BEV_CHANNELS, nx, ny)
    bev_lid = torch.randn(LIDAR_BEV_CHANNELS, nx, ny)
    det = BEVDetector.from_bev_maps(bev_cam, bev_lid, num_classes=3).eval()
    describe(det)
    with torch.no_grad():
        out = det(bev_cam, bev_lid)
    print({k: tuple(v.shape) for k, v in out.items()})
