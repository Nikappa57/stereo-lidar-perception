"""Evaluation and post-processing for the BEV perception pipeline."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import globals as G


class CenterPointDecoder:
    """Decodes dense CenterPoint heatmaps and offsets into 2D object lists.
    
    Uses 3x3 max-pooling as a fast, hardware-friendly Non-Maximum Suppression (NMS)
    to find local peaks in the heatmap.
    """

    def __init__(self, score_threshold: float = 0.2, max_objects: int = 100):
        self.score_threshold = score_threshold
        self.max_objects = max_objects

        # Grid geometry imported directly from globals.py
        # to ensure the mapping matches the one used during splatting.
        self.x_min = G.X_RANGE[0]
        self.y_min = G.Y_RANGE[0]
        self.res = G.BEV_RES_M

    def __call__(self, heatmap: torch.Tensor,
                 offset: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """
        Args:
            heatmap: (B, num_classes, nx, ny) - Raw logits from the head.
            offset:  (B, 2, nx, ny) - Continuous sub-pixel corrections.
            
        Returns:
            List of B dicts. Each dict contains:
                'boxes_2d': (N, 2) [x_metric, y_metric]
                'scores':   (N,) confidences
                'classes':  (N,) class indices
        """
        B, C, nx, ny = heatmap.shape

        # 1. Convert logits to probabilities [0, 1]
        scores = torch.sigmoid(heatmap)

        # 2. Hardware-friendly NMS (Max Pooling 3x3)
        # Replaces each cell's value with the maximum in its 3x3 neighborhood.
        # If a cell matches the local maximum, its probability remains unchanged.
        # Otherwise, it is masked to zero.
        max_scores = F.max_pool2d(scores, kernel_size=3, stride=1, padding=1)
        peak_mask = (max_scores == scores)
        scores = scores * peak_mask.float()

        results = []
        for b in range(B):
            # Flatten spatial tensors for easier index extraction
            # scores_b: (C, nx*ny)
            scores_b = scores[b].reshape(C, -1)
            offset_b = offset[b].reshape(2, -1)

            # Find values exceeding the threshold
            valid_mask = scores_b > self.score_threshold

            # Indices of valid candidates.
            # idx_c: class, idx_flat: flat 1D index in the grid
            idx_c, idx_flat = valid_mask.nonzero(as_tuple=True)

            val_scores = scores_b[idx_c, idx_flat]

            # Apply maximum object limit per frame (e.g., top 100)
            if len(val_scores) > self.max_objects:
                val_scores, topk_idx = torch.topk(val_scores, self.max_objects)
                idx_c = idx_c[topk_idx]
                idx_flat = idx_flat[topk_idx]

            # Reconstruct grid coordinates (ix, iy)
            ix = torch.div(idx_flat, ny, rounding_mode='trunc')
            iy = idx_flat % ny

            # Extract sub-pixel offsets for selected cells
            dx = offset_b[0, idx_flat]
            dy = offset_b[1, idx_flat]

            # Continuous coordinates on the grid
            grid_x = ix.float() + dx
            grid_y = iy.float() + dy

            # Inverse mapping: Grid coordinates -> Metric coordinates (Ego frame)
            metric_x = grid_x * self.res + self.x_min
            metric_y = grid_y * self.res + self.y_min

            boxes_2d = torch.stack([metric_x, metric_y], dim=-1)

            results.append({
                "boxes_2d": boxes_2d,
                "scores": val_scores,
                "classes": idx_c
            })

        return results


# =========================================================================== #
# Distance-AP evaluation (AV2 style: TP if centre within d metres of a GT)
# =========================================================================== #
def frame_ground_truth(sample) -> tuple[np.ndarray, np.ndarray]:
    """Evaluable GT for one frame: in-grid, class-remapped centres.

    Applies the same filter as the target encoder (:func:`train.encode_sample`):
    labels outside :data:`globals.CLASSES` are ignored, centres outside the BEV
    grid are not evaluable (the network cannot see them).

    :returns: ``(centres (M, 2) float64 ego xy, classes (M,) int64)``.
    """
    boxes = np.asarray(sample.boxes_3d_ego, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)
    idx = [G.class_index(l) for l in sample.boxes_3d_labels]
    x, y = boxes[:, 0], boxes[:, 1]
    keep = np.array([c is not None for c in idx]) & \
        (x >= G.X_RANGE[0]) & (x < G.X_RANGE[1]) & \
        (y >= G.Y_RANGE[0]) & (y < G.Y_RANGE[1])
    centres = boxes[keep, :2]
    classes = np.array([c for c, k in zip(idx, keep) if k], dtype=np.int64)
    return centres, classes


_EMPTY_OP = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "score": float("nan")}
_EMPTY_CURVE = {"precision": np.zeros(0), "recall": np.zeros(0),
                "scores": np.zeros(0)}


def _average_precision(scores, xy, frame_ids, gt_by_frame, n_gt,
                       threshold: float
                       ) -> tuple[float, np.ndarray, dict, dict]:
    """AP + F1-optimal operating point + raw PR curve, one class/threshold.

    Predictions are visited in descending score order; each may claim the
    nearest *unmatched* GT of its frame within ``threshold`` metres (TP),
    otherwise it is a FP. AP is all-point interpolated (precision envelope);
    the operating point is the position on the *raw* PR curve that maximises
    F1, i.e. "keep detections with confidence >= op['score']" gives
    ``op['precision']`` / ``op['recall']`` / ``op['f1']``.

    :param gt_by_frame: ``{frame_id: (M_f, 2) gt centres}`` for this class.
    :param n_gt: Total GT count for this class (recall denominator).
    :returns: ``(ap, tp_distances, op, curve)`` — matched centre errors feed
        the mean-error metric; ``curve`` holds the raw ``precision`` /
        ``recall`` / ``scores`` arrays (score-descending) for plotting.
    """
    if n_gt == 0:
        return float("nan"), np.zeros(0), dict(_EMPTY_OP), dict(_EMPTY_CURVE)
    if len(scores) == 0:
        return 0.0, np.zeros(0), dict(_EMPTY_OP), dict(_EMPTY_CURVE)

    order = np.argsort(-scores)
    matched = {f: np.zeros(len(g), dtype=bool) for f, g in gt_by_frame.items()}
    tp = np.zeros(len(order))
    tp_dist = []
    for rank, i in enumerate(order):
        gts = gt_by_frame.get(int(frame_ids[i]))
        if gts is None or len(gts) == 0:
            continue  # tp stays 0 -> counted as FP below
        d = np.linalg.norm(gts - xy[i], axis=1)
        d[matched[int(frame_ids[i])]] = np.inf
        j = int(d.argmin())
        if d[j] < threshold:
            tp[rank] = 1.0
            matched[int(frame_ids[i])][j] = True
            tp_dist.append(float(d[j]))

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # F1-optimal operating point on the raw curve (before the AP envelope)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(f1.argmax())
    op = {"precision": float(precision[best]), "recall": float(recall[best]),
          "f1": float(f1[best]), "score": float(scores[order[best]])}
    curve = {"precision": precision.copy(), "recall": recall.copy(),
             "scores": np.asarray(scores)[order]}

    # precision envelope (monotone non-increasing), then integrate over recall
    precision = precision.copy()
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
    return ap, np.asarray(tp_dist), op, curve


def evaluate_model(model, frames, *, input_fn=None,
                   device: torch.device = torch.device("cpu"),
                   score_threshold: float = 0.1,
                   sample_kwargs: dict | None = None,
                   thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)) -> dict:
    """AV2-style distance-AP of a detector over a list of :class:`~data.Frame`.

    Runs the model + :class:`CenterPointDecoder` on every frame, then per class
    computes AP at each distance threshold (0.5/1/2/4 m, the AV2 bands) and
    their mean; ``mAP`` averages the per-class means over classes that have GT.
    ``mean_error_m`` is the mean centre error of the matches at the 2 m band.
    (CDS — the size/orientation-aware composite — needs boxes we don't regress;
    TODO with the full AV2 toolkit integration.)

    :param input_fn: ``sample -> model input``; identity (Pipeline) by default,
        ``network.lidar_points`` for the LiDAR-only baseline.
    :returns: ``{"per_class": {name: {thr: ap, "mean": m, "n_gt": n}},
        "mAP": float, "mean_error_m": float, "n_frames": int}``
    """
    input_fn = input_fn or (lambda s: s)
    # fast sample assembly: no images / point mask (same as train.FAST_SAMPLE_KWARGS);
    # the camera branch reads the precomputed stereo cache. Pass {} for full samples.
    if sample_kwargs is None:
        sample_kwargs = {"load_images": False, "point_mask": False}
    decoder = CenterPointDecoder(score_threshold=score_threshold)

    # scores/xy/frame ids per class + GT centres per (class, frame)
    preds: dict[int, dict[str, list]] = {
        c: {"scores": [], "xy": [], "frame": []} for c in range(G.NUM_CLASSES)}
    gt_by_class: dict[int, dict[int, np.ndarray]] = {
        c: {} for c in range(G.NUM_CLASSES)}
    n_gt = np.zeros(G.NUM_CLASSES, dtype=np.int64)

    frame_dets: list[dict] = []          # raw per-frame preds (for confusion)
    frame_gts: list[tuple] = []          # raw per-frame GT (for confusion)
    model.eval()
    with torch.no_grad():
        for f_id, frame in enumerate(frames):
            sample = frame.to_stereo_sample(**sample_kwargs)
            out = model(input_fn(sample), device=device)
            det = decoder(out["heatmap"].cpu(), out["offset"].cpu())[0]
            for c in range(G.NUM_CLASSES):
                m = det["classes"] == c
                preds[c]["scores"].extend(det["scores"][m].tolist())
                preds[c]["xy"].extend(det["boxes_2d"][m].tolist())
                preds[c]["frame"].extend([f_id] * int(m.sum()))
            centres, classes = frame_ground_truth(sample)
            for c in range(G.NUM_CLASSES):
                gt_by_class[c][f_id] = centres[classes == c]
                n_gt[c] += int((classes == c).sum())
            frame_dets.append({"xy": det["boxes_2d"].numpy(),
                               "scores": det["scores"].numpy(),
                               "classes": det["classes"].numpy()})
            frame_gts.append((centres, classes))

    # the distance band used for the P/R/F1 operating point + mean error
    op_thr = 2.0 if 2.0 in thresholds else thresholds[len(thresholds) // 2]

    report: dict = {"per_class": {}, "n_frames": len(frames),
                    "op_threshold_m": op_thr}
    class_means, errors = [], []
    for c, name in enumerate(G.CLASSES):
        scores = np.asarray(preds[c]["scores"])
        xy = np.asarray(preds[c]["xy"]).reshape(-1, 2)
        f_ids = np.asarray(preds[c]["frame"], dtype=np.int64)
        row: dict = {"n_gt": int(n_gt[c]), "ops": {}, "curves": {}}
        aps = []
        for thr in thresholds:
            ap, tp_dist, op, curve = _average_precision(scores, xy, f_ids,
                                                        gt_by_class[c],
                                                        int(n_gt[c]), thr)
            row[thr] = ap
            row["ops"][thr] = op          # F1-optimal precision/recall/f1/score
            row["curves"][thr] = curve    # raw PR curve arrays (plotting)
            aps.append(ap)
            if thr == op_thr:
                errors.extend(tp_dist.tolist())
        row["mean"] = float(np.nanmean(aps)) if n_gt[c] else float("nan")
        report["per_class"][name] = row
        if n_gt[c]:
            class_means.append(row["mean"])
    report["mAP"] = float(np.mean(class_means)) if class_means else float("nan")
    report["mean_error_m"] = float(np.mean(errors)) if errors else float("nan")

    # macro P/R/F1 at op_thr over classes that have GT
    ops = [report["per_class"][n]["ops"][op_thr]
           for c, n in enumerate(G.CLASSES) if n_gt[c]]
    for key in ("precision", "recall", "f1"):
        report[key] = float(np.mean([o[key] for o in ops])) if ops else float("nan")

    # confusion matrix at op_thr, using each class's F1-optimal confidence
    # cutoff (0.3 fallback for classes with no GT, so their FPs still show)
    cutoffs = np.array([
        report["per_class"][name]["ops"][op_thr]["score"]
        for name in G.CLASSES])
    cutoffs = np.where(np.isfinite(cutoffs), cutoffs, 0.3)
    report["confusion"] = {
        "matrix": _confusion_matrix(frame_dets, frame_gts, op_thr, cutoffs),
        "labels": tuple(G.CLASSES) + ("background",),
        "dist_threshold_m": op_thr,
        "score_cutoffs": cutoffs.tolist(),
    }
    return report


def _confusion_matrix(frame_dets, frame_gts, dist_thr: float,
                      score_cutoffs: np.ndarray) -> np.ndarray:
    """Detection confusion matrix: rows = GT class, cols = predicted class.

    Detections below their class's confidence cutoff are dropped; the rest are
    matched **class-agnostically** (best score first, nearest unmatched GT
    within ``dist_thr``), so cross-class confusions land off-diagonal. The
    extra last row/column is "background": unmatched detections (FPs) fill the
    background *row*, missed GTs (FNs) fill the background *column*.
    """
    C = G.NUM_CLASSES
    m = np.zeros((C + 1, C + 1), dtype=np.int64)
    for det, (gt_xy, gt_cls) in zip(frame_dets, frame_gts):
        keep = det["scores"] >= score_cutoffs[det["classes"]]
        xy = det["xy"][keep]
        cls = det["classes"][keep]
        order = np.argsort(-det["scores"][keep])
        matched = np.zeros(len(gt_xy), dtype=bool)
        for i in order:
            if len(gt_xy):
                d = np.linalg.norm(gt_xy - xy[i], axis=1)
                d[matched] = np.inf
                j = int(d.argmin())
            else:
                j, d = 0, np.array([np.inf])
            if len(gt_xy) and d[j] < dist_thr:
                m[int(gt_cls[j]), int(cls[i])] += 1
                matched[j] = True
            else:
                m[C, int(cls[i])] += 1          # FP: no GT nearby
        for j in np.flatnonzero(~matched):
            m[int(gt_cls[j]), C] += 1           # FN: GT never claimed
    return m


def print_ap_report(report: dict) -> None:
    """Pretty-print an :func:`evaluate_model` report: AP table + P/R/F1 table."""
    thrs = [k for k in next(iter(report["per_class"].values())) if isinstance(k, float)]
    header = "class".ljust(14) + "".join(f"AP@{t:<5g}" for t in thrs) + "  mean   n_gt"
    print(header)
    print("-" * len(header))
    for name, row in report["per_class"].items():
        cells = "".join(f"{row[t]:<8.3f}" if row["n_gt"] else f"{'—':<8s}" for t in thrs)
        mean = f"{row['mean']:<7.3f}" if row["n_gt"] else f"{'—':<7s}"
        print(f"{name:<14s}{cells}{mean}{row['n_gt']}")

    op_thr = report["op_threshold_m"]
    print(f"\nF1-optimal operating point @{op_thr:g} m "
          f"(apply 'confidence >= score' at deployment):")
    print(f"{'class':<14s}{'prec':<8s}{'recall':<8s}{'F1':<8s}{'score':<8s}")
    print("-" * 46)
    for name, row in report["per_class"].items():
        op = row["ops"][op_thr]
        if row["n_gt"]:
            print(f"{name:<14s}{op['precision']:<8.3f}{op['recall']:<8.3f}"
                  f"{op['f1']:<8.3f}{op['score']:<8.3f}")
        else:
            print(f"{name:<14s}{'—':<8s}{'—':<8s}{'—':<8s}{'—':<8s}")

    print(f"\nmAP {report['mAP']:.3f} | macro P {report['precision']:.3f} "
          f"R {report['recall']:.3f} F1 {report['f1']:.3f} @{op_thr:g} m | "
          f"mean centre error (TP@{op_thr:g}m) {report['mean_error_m']:.3f} m | "
          f"{report['n_frames']} frames")


def save_report(report: dict, path: str | Path) -> Path:
    """Save an :func:`evaluate_model` report to JSON (numpy → lists).

    One file per run (e.g. ``results/lidar.json``) so trained approaches can be
    compared later with :func:`compare_reports` without re-running evaluation.
    """
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"not JSON-serializable: {type(o)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, default=default, indent=1))
    return path


def load_report(path: str | Path) -> dict:
    """Load a :func:`save_report` JSON, restoring the in-memory layout.

    JSON turns the float threshold keys into strings and arrays into lists;
    this puts them back so ``print_ap_report`` / ``visualize_evaluation`` work
    on a loaded report exactly as on a fresh one.
    """
    report = json.loads(Path(path).read_text())
    for row in report["per_class"].values():
        for key in [k for k in row if k not in ("n_gt", "mean", "ops", "curves")]:
            row[float(key)] = row.pop(key)
        row["ops"] = {float(k): v for k, v in row["ops"].items()}
        row["curves"] = {
            float(k): {n: np.asarray(a) for n, a in c.items()}
            for k, c in row["curves"].items()}
    conf = report.get("confusion")
    if conf is not None:
        conf["matrix"] = np.asarray(conf["matrix"], dtype=np.int64)
        conf["labels"] = tuple(conf["labels"])
    return report


def compare_reports(reports: dict[str, dict]) -> None:
    """Side-by-side table of saved runs: ``{run_name: report}``.

    Per-class columns are the mean AP over the distance thresholds; P/R/F1 are
    the macro values at each report's operating threshold (2 m by default).
    """
    name_w = max(16, *(len(n) for n in reports)) + 2  # fit the longest run name
    header = ("model".ljust(name_w)
              + "".join(f"{n[:10]:<11s}" for n in G.CLASSES)
              + f"{'mAP':<7s}{'P':<7s}{'R':<7s}{'F1':<7s}err(m)")
    print(header)
    print("-" * len(header))
    for name, rep in reports.items():
        cells = "".join(
            f"{rep['per_class'][c]['mean']:<11.3f}"
            if rep["per_class"][c]["n_gt"] else f"{'—':<11s}"
            for c in G.CLASSES)
        print(f"{name:<{name_w}s}{cells}{rep['mAP']:<7.3f}{rep['precision']:<7.3f}"
              f"{rep['recall']:<7.3f}{rep['f1']:<7.3f}{rep['mean_error_m']:.3f}")
