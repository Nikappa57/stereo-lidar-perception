"""Target generation, loss and training loop for BEV object detection."""

import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import globals as G

# Column layout of boxes_3d_ego (BoundingBoxSE3Index, see data.py):
#   0:x 1:y 2:z  3:qw 4:qx 5:qy 6:qz  7:length 8:width 9:height
_X, _Y = 0, 1
_LENGTH, _WIDTH = 7, 8


def gaussian_radius(det_size: tuple[float, float],
                    min_overlap: float = 0.7) -> float:
    """CornerNet/CenterPoint Gaussian radius (in cells).

    The largest radius such that a predicted box equal to ``det_size``
    ``(height, width)`` in cells, but shifted by that radius, still overlaps the
    GT box with IoU >= ``min_overlap``. Solves the three shift cases (inside /
    outside / straddling) and takes the tightest. Larger objects → larger
    radius; small ones get a sensible non-trivial spread instead of one cell.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / (2 * a3)

    return min(r1, r2, r3)


class TargetEncoder:
    """Generates ground truth heatmaps and offsets from 3D bounding boxes.

    Axis convention (shared with the splat / scatter / decoder): heatmap and
    offset are ``(*, nx, ny)`` with **dim 0 = x** (forward, ``nx``) and
    **dim 1 = y** (lateral, ``ny``); the flat index is ``ix * ny + iy``.
    """

    def __init__(self, num_classes: int = G.NUM_CLASSES,
                 radius_mode: str = "iou",
                 min_overlap: float = 0.1, min_radius: int = 2):
        # radius_mode:
        #   "iou" (default) — CornerNet/CenterPoint gaussian_radius at
        #     `min_overlap` (=0.1 + min_radius=2, the mmdet3d BEV convention).
        #     The tighter blob keeps near-centre negatives supervised, so the
        #     decoded peak stays sharp — matters for distance-AP @0.5 m.
        #   "proportional" — radius = max(length, width)/2 in cells, so the blob
        #     roughly covers the footprint (big cars ~9 cells → flat top, peak
        #     can wander ±1-2 cells). Kept as an ablation option.
        self.num_classes = num_classes
        self.radius_mode = radius_mode
        self.min_overlap = min_overlap  # IoU floor for gaussian_radius
        self.min_radius = min_radius    # never shrink below this many cells
        self.x_min = G.X_RANGE[0]
        self.y_min = G.Y_RANGE[0]
        self.res = G.BEV_RES_M
        self.nx = int((G.X_RANGE[1] - G.X_RANGE[0]) / G.BEV_RES_M)
        self.ny = int((G.Y_RANGE[1] - G.Y_RANGE[0]) / G.BEV_RES_M)

    def _draw_gaussian(self, heatmap, center, radius):
        """Draw a 2D Gaussian into ``heatmap`` (shape ``(nx, ny)``) peaked at
        ``center = (ix, iy)``. dim 0 is x (``nx``), dim 1 is y (``ny``)."""
        ix, iy = center
        sigma = (2 * radius + 1) / 6.0  # CenterNet draw_umich convention

        x_lo = int(max(0, ix - radius))
        x_hi = int(min(self.nx, ix + radius + 1))
        y_lo = int(max(0, iy - radius))
        y_hi = int(min(self.ny, iy + radius + 1))

        xs, ys = torch.meshgrid(torch.arange(x_lo, x_hi),
                                torch.arange(y_lo, y_hi),
                                indexing='ij')
        dist_sq = (xs - ix)**2 + (ys - iy)**2
        g = torch.exp(-dist_sq.float() / (2 * sigma**2))
        heatmap[x_lo:x_hi, y_lo:y_hi] = torch.max(heatmap[x_lo:x_hi, y_lo:y_hi],
                                                  g)

    def encode(self, boxes_3d, classes):
        """
        Args:
            boxes_3d: (N, 10) [x, y, z, qw, qx, qy, qz, length, width, height]
            classes:  (N,) class indices, each in [0, num_classes)

        Returns:
            heatmap: (num_classes, nx, ny)
            offset:  (2, nx, ny)
        """
        heatmap = torch.zeros((self.num_classes, self.nx, self.ny))
        offset = torch.zeros((2, self.nx, self.ny))

        for i in range(len(boxes_3d)):
            # 1. centre in continuous grid coordinates
            gx = (boxes_3d[i, _X] - self.x_min) / self.res
            gy = (boxes_3d[i, _Y] - self.y_min) / self.res

            # 2. Gaussian radius (cells) from the box footprint
            l_cells = float(boxes_3d[i, _LENGTH]) / self.res
            w_cells = float(boxes_3d[i, _WIDTH]) / self.res
            if self.radius_mode == "iou":
                radius = gaussian_radius((l_cells, w_cells), self.min_overlap)
            else:  # "proportional": blob ~ covers the footprint
                radius = max(l_cells, w_cells) / 2.0
            radius_cells = max(self.min_radius, int(round(radius)))

            # floor, not int(): int() truncates toward zero, so a centre just
            # below the grid edge (gx = -0.4) would alias into cell 0 with a
            # negative offset instead of being rejected by the bounds check.
            ix, iy = math.floor(gx), math.floor(gy)
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                cls = int(classes[i])
                # 3. Gaussian peak on this class' channel
                self._draw_gaussian(heatmap[cls], (ix, iy), radius_cells)
                # 4. sub-cell offset of the true centre
                offset[0, ix, iy] = gx - ix
                offset[1, ix, iy] = gy - iy

        return heatmap, offset


# =========================================================================== #
# Loss — CenterPoint 2D: Gaussian-focal heatmap + masked L1 centre offset
# =========================================================================== #
def gaussian_focal_loss(pred: torch.Tensor, target: torch.Tensor,
                        alpha: float = 2.0, beta: float = 4.0,
                        eps: float = 1e-4) -> torch.Tensor:
    """Penalty-reduced ('Gaussian') focal loss for CenterNet heatmaps.

    ``pred`` are probabilities in (0, 1); ``target`` is the Gaussian heatmap in
    [0, 1] with an exact 1.0 at each object centre. Both ``(B, C, H, W)``.
    Only exact-centre cells are positives; every other cell is a negative whose
    penalty is scaled by ``(1 - target)**beta``, so cells near a centre barely
    contribute. Normalised by the number of positives (CenterNet convention).
    """
    pred = pred.clamp(eps, 1 - eps)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    neg_weight = (1 - target).pow(beta)

    pos_loss = torch.log(pred) * (1 - pred).pow(alpha) * pos
    neg_loss = torch.log(1 - pred) * pred.pow(alpha) * neg_weight * neg

    n_pos = pos.sum()
    pos_loss, neg_loss = pos_loss.sum(), neg_loss.sum()
    if n_pos == 0:  # no objects in view → only the (down-weighted) negatives
        return -neg_loss
    return -(pos_loss + neg_loss) / n_pos


class CenterPointLoss(nn.Module):
    """CenterPoint 2D loss: Gaussian-focal heatmap + masked L1 centre offset.

    Classification is *implicit* — one heatmap channel per class — so there is
    no separate cross-entropy term. ``pred`` is the head's output dict of
    **logits** (``heatmap`` + ``offset``); ``target_heatmap`` / ``target_offset``
    are the :class:`TargetEncoder` tensors with a leading batch dim.

    The offset is regressed **only at positive centre cells** (an empty cell has
    no centre to correct), masked from ``target_heatmap == 1``.
    """

    def __init__(self, offset_weight: float = 1.0, alpha: float = 2.0,
                 beta: float = 4.0):
        super().__init__()
        self.offset_weight = offset_weight
        self.alpha, self.beta = alpha, beta

    def forward(self, pred: dict[str, torch.Tensor],
                target_heatmap: torch.Tensor,
                target_offset: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, float]]:
        hm = pred["heatmap"].sigmoid()
        hm_loss = gaussian_focal_loss(hm, target_heatmap, self.alpha, self.beta)

        # positive (centre) cells: any class channel at its exact peak
        pos = target_heatmap.eq(1).any(dim=1, keepdim=True).float()  # (B,1,H,W)
        off_l1 = F.l1_loss(pred["offset"], target_offset, reduction="none")
        off_loss = (off_l1 * pos).sum() / pos.sum().clamp(min=1.0)

        total = hm_loss + self.offset_weight * off_loss
        return total, {"heatmap": float(hm_loss), "offset": float(off_loss)}


# =========================================================================== #
# Single-frame overfit loop (P1 sanity harness)
# =========================================================================== #
# The trainable assemblies live in network.py: LidarOnlyDetector (baseline),
# PipelineA / PipelineB / PipelineC (branches + fusion + head, end-to-end).
def encode_sample(sample, encoder: "TargetEncoder"
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """StereoSample → batched ``(heatmap, offset)`` targets.

    Applies the globals class remap (dropping ignored classes) and the BEV-grid
    filter, then encodes — the same recipe the consistency test/visualiser use.
    """
    boxes = torch.as_tensor(sample.boxes_3d_ego, dtype=torch.float32)
    idx = [G.class_index(l) for l in sample.boxes_3d_labels]
    if len(boxes) == 0:  # frames with no GT (common on KITTI-360, sparse labels)
        return (torch.zeros((1, G.NUM_CLASSES, encoder.nx, encoder.ny)),
                torch.zeros((1, 2, encoder.nx, encoder.ny)))
    x, y = boxes[:, 0], boxes[:, 1]
    inside = ((x >= G.X_RANGE[0]) & (x < G.X_RANGE[1]) &
              (y >= G.Y_RANGE[0]) & (y < G.Y_RANGE[1]))
    # dtype=bool: an empty list would otherwise make a Float tensor and break `&`.
    keep = torch.tensor([c is not None for c in idx], dtype=torch.bool) & inside
    bx = boxes[keep]
    lab = torch.tensor([c for c, k in zip(idx, keep.tolist()) if k],
                       dtype=torch.long)
    hm, off = encoder.encode(bx, lab)
    return hm.unsqueeze(0), off.unsqueeze(0)


def overfit_one_frame(model: nn.Module, inputs,
                      target_heatmap: torch.Tensor, target_offset: torch.Tensor,
                      steps: int = 150, lr: float = 1e-3,
                      device: torch.device = torch.device("cpu")
                      ) -> list[float]:
    """Overfit a single frame — the sanity check that the whole loop learns.

    ``inputs`` is whatever ``model`` consumes: the ``(N, 4)`` point array for
    :class:`network.LidarOnlyDetector`, the whole ``StereoSample`` for the
    :class:`network.Pipeline` classes.

    Returns the per-step total-loss history. A healthy loop drives it steadily
    down; decoding the final prediction should then land on the GT centres.
    """
    model.to(device).train()
    loss_fn = CenterPointLoss()
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    tgt_hm, tgt_off = target_heatmap.to(device), target_offset.to(device)

    history = []
    for _ in range(steps):
        opt.zero_grad()
        pred = model(inputs, device=device)
        loss, _ = loss_fn(pred, tgt_hm, tgt_off)
        loss.backward()
        opt.step()
        history.append(float(loss))
    return history


# =========================================================================== #
# Multi-frame training loop (P1)
# =========================================================================== #
def set_seed(seed: int = 0) -> None:
    """Seed every RNG that touches training (python, numpy, torch, CUDA).

    Makes runs *reproducible*, not bit-deterministic: the BEV splat and other
    scatter-style CUDA kernels use atomics whose accumulation order varies, so
    losses can still differ in the last decimals between identical runs. For
    strict determinism add ``torch.use_deterministic_algorithms(True)`` (slower,
    and some ops may raise).
    """
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # seeds CUDA too (all devices)
    torch.backends.cudnn.benchmark = False  # don't let autotune pick per-run kernels


def split_frames(dataset, val_scenes: "int | list[int]" = 1) -> tuple[list, list]:
    """Split a :class:`~data.Py123dDataset` into train/val **by scene (log)**.

    Frames within one log are strongly correlated (consecutive timesteps of the
    same drive), so a per-frame split would leak train scenes into val.

    :param val_scenes: ``int`` — hold out the *last* n logs; or an explicit
        list of scene indices to hold out (e.g. ``[1]`` to validate on a log
        that actually contains the rare class you care about).
    :returns: ``(train_frames, val_frames)`` — lists of :class:`~data.Frame`.
    """
    if isinstance(val_scenes, int):
        assert 0 < val_scenes < dataset.scene_count, (
            f"val_scenes={val_scenes} must leave at least one scene on each "
            f"side of the split (dataset has {dataset.scene_count})")
        val_ids = set(range(dataset.scene_count - val_scenes,
                            dataset.scene_count))
    else:
        val_ids = set(int(i) for i in val_scenes)
        assert val_ids and all(0 <= i < dataset.scene_count for i in val_ids), (
            f"val scene indices {sorted(val_ids)} out of range "
            f"(dataset has {dataset.scene_count} scenes)")
        assert len(val_ids) < dataset.scene_count, "no scenes left for training"
    train_frames, val_frames = [], []
    for scene_index in range(dataset.scene_count):
        bucket = val_frames if scene_index in val_ids else train_frames
        bucket.extend(dataset.frames_in_scene(scene_index))
    return train_frames, val_frames


# Training needs neither the images (LiDAR path; the camera branch reads the
# precomputed stereo cache) nor the LiDAR-in-box mask — skipping both cuts
# sample assembly from ~0.9 s to ~30 ms (see to_stereo_sample docstring).
FAST_SAMPLE_KWARGS = {"load_images": False, "point_mask": False}


def validate(model: nn.Module, frames, *, input_fn=None,
             encoder: "TargetEncoder | None" = None,
             sample_kwargs: dict | None = None,
             device: torch.device = torch.device("cpu")) -> float:
    """Mean :class:`CenterPointLoss` over ``frames`` (no grad, eval mode)."""
    input_fn = input_fn or (lambda s: s)
    encoder = encoder or TargetEncoder()
    sample_kwargs = FAST_SAMPLE_KWARGS if sample_kwargs is None else sample_kwargs
    loss_fn = CenterPointLoss()
    model.eval()
    losses = []
    with torch.no_grad():
        for frame in frames:
            sample = frame.to_stereo_sample(**sample_kwargs)
            tgt_hm, tgt_off = encode_sample(sample, encoder)
            pred = model(input_fn(sample), device=device)
            loss, _ = loss_fn(pred, tgt_hm.to(device), tgt_off.to(device))
            losses.append(float(loss))
    return sum(losses) / max(1, len(losses))


def train_model(model: nn.Module, train_frames, val_frames, *,
                input_fn=None, epochs: int = 5, lr: float = 1e-3,
                accum: int = 4, encoder: "TargetEncoder | None" = None,
                ckpt_path: str | Path | None = None, log_every: int = 50,
                seed: int = 0, sample_kwargs: dict | None = None,
                device: torch.device = torch.device("cpu")) -> dict:
    """Multi-frame training loop (P1): frame-by-frame + gradient accumulation.

    The branches process one sample at a time (batch dim 1), so an effective
    batch is built by accumulating ``accum`` frames' gradients per optimizer
    step. Frames are re-shuffled every epoch; validation runs after each epoch
    and the best-val checkpoint is written to ``ckpt_path`` (if given).

    CPU/GPU overlap: the next frame's sample assembly + target encoding runs
    in a background thread while the GPU steps on the current one, and samples
    are loaded with :data:`FAST_SAMPLE_KWARGS` (no images, no point mask) —
    the camera branch must therefore read the precomputed stereo cache
    (:func:`data.precompute_stereo_inputs`); pass ``sample_kwargs={}`` to
    force full samples instead.

    :param input_fn: ``sample -> model input``; identity for the
        :class:`network.Pipeline` classes (default), ``network.lidar_points``
        for :class:`network.LidarOnlyDetector`.
    :returns: history dict — per-epoch ``"train"`` / ``"val"`` mean losses and
        the per-step ``"steps"`` list (for plotting).
    """
    from concurrent.futures import ThreadPoolExecutor

    input_fn = input_fn or (lambda s: s)
    encoder = encoder or TargetEncoder()
    sample_kwargs = FAST_SAMPLE_KWARGS if sample_kwargs is None else sample_kwargs
    loss_fn = CenterPointLoss()
    model.to(device)
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    rng = random.Random(seed)
    order = list(range(len(train_frames)))

    def _prep(frame):
        sample = frame.to_stereo_sample(**sample_kwargs)
        tgt_hm, tgt_off = encode_sample(sample, encoder)
        return input_fn(sample), tgt_hm, tgt_off

    history: dict = {"train": [], "val": [], "steps": []}
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        opt.zero_grad()
        total = 0.0
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_prep, train_frames[order[0]])
            for k in range(len(order)):
                inputs, tgt_hm, tgt_off = future.result()
                if k + 1 < len(order):  # prefetch the next frame
                    future = pool.submit(_prep, train_frames[order[k + 1]])
                pred = model(inputs, device=device)
                loss, _ = loss_fn(pred, tgt_hm.to(device), tgt_off.to(device))
                (loss / accum).backward()
                if (k + 1) % accum == 0 or k == len(order) - 1:
                    opt.step()
                    opt.zero_grad()
                total += float(loss)
                history["steps"].append(float(loss))
                if log_every and (k + 1) % log_every == 0:
                    print(f"  epoch {epoch} step {k + 1}/{len(order)} "
                          f"loss {float(loss):.3f}")

        train_mean = total / max(1, len(order))
        val_mean = validate(model, val_frames, input_fn=input_fn,
                            encoder=encoder, sample_kwargs=sample_kwargs,
                            device=device)
        history["train"].append(train_mean)
        history["val"].append(val_mean)
        print(f"epoch {epoch}/{epochs}  train {train_mean:.3f}  "
              f"val {val_mean:.3f}")

        if ckpt_path is not None and val_mean < best_val:
            best_val = val_mean
            ckpt = Path(ckpt_path)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_mean}, ckpt)
            print(f"  new best val — checkpoint saved → {ckpt}")
    return history
