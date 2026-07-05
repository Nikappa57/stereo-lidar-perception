"""Encoder–decoder consistency: every in-grid GT centre survives encode→decode."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import globals as G  # noqa: E402
from data import Py123dDataset  # noqa: E402
from evaluation import CenterPointDecoder  # noqa: E402
from train import TargetEncoder  # noqa: E402


def test_encoder_decoder_consistency():
    # 1. Load a real sample
    dataset = Py123dDataset(split_names=["kitti360_train"])
    sample = dataset[0].to_stereo_sample()

    boxes = torch.as_tensor(sample.boxes_3d_ego, dtype=torch.float32)  # (N, 10)

    # Map loader labels to training class indices; drop ignored classes (None)
    # and boxes whose centre is outside the BEV grid — the encoder rasterises
    # neither (legitimate labels also sit behind/beside the crop).
    idx = [G.class_index(l) for l in sample.boxes_3d_labels]
    x, y = boxes[:, 0], boxes[:, 1]
    inside = ((x >= G.X_RANGE[0]) & (x < G.X_RANGE[1]) &
              (y >= G.Y_RANGE[0]) & (y < G.Y_RANGE[1]))
    keep = torch.tensor([c is not None for c in idx]) & inside
    boxes = boxes[keep]
    labels = torch.tensor([c for c, k in zip(idx, keep.tolist()) if k],
                          dtype=torch.long)
    assert len(boxes) > 0, "no kept GT boxes inside the BEV grid for this frame"

    # 2. Encoder: GT -> heatmap + offset
    encoder = TargetEncoder(num_classes=G.NUM_CLASSES)
    heatmap, offset = encoder.encode(boxes, labels)

    # 3. Decoder: heatmap -> predictions
    decoder = CenterPointDecoder(score_threshold=0.5)
    preds = decoder(heatmap.unsqueeze(0), offset.unsqueeze(0))
    decoded_boxes = preds[0]["boxes_2d"]
    assert len(decoded_boxes) > 0, "decoder produced no detections"

    # 4. Each in-grid GT centre must be recovered. The continuous sub-cell offset
    #    is stored, so recovery is near-exact; one cell of slack covers the rare
    #    case of two boxes sharing a cell.
    tolerance = G.BEV_RES_M
    for i in range(len(boxes)):
        gt = boxes[i, :2]
        min_dist = torch.norm(decoded_boxes - gt, dim=1).min()
        assert min_dist < tolerance, (
            f"GT box {i} at {gt.tolist()} not recovered (dist={min_dist:.3f} m)")

    print(f"\nRecovered {len(boxes)} in-grid GT centres "
          f"from {len(decoded_boxes)} decoded peaks.")
