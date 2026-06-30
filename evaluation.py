"""Evaluation and post-processing for the BEV perception pipeline."""

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
