"""Target generation utilities for BEV object detection training."""

import math

import torch

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
                 radius_mode: str = "proportional",
                 min_overlap: float = 0.1, min_radius: int = 2):
        # radius_mode:
        #   "proportional" — radius = max(length, width)/2 in cells, so the blob
        #     roughly covers the footprint (big cars, small cones). Best on this
        #     fine 0.25 m grid where the IoU radius saturates (~3 cells for a car
        #     regardless of min_overlap).
        #   "iou" — CornerNet/CenterPoint gaussian_radius at `min_overlap`
        #     (=0.1, the mmdet3d BEV convention; 0.7 collapses to min_radius here).
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

            ix, iy = int(gx), int(gy)
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                cls = int(classes[i])
                # 3. Gaussian peak on this class' channel
                self._draw_gaussian(heatmap[cls], (ix, iy), radius_cells)
                # 4. sub-cell offset of the true centre
                offset[0, ix, iy] = gx - ix
                offset[1, ix, iy] = gy - iy

        return heatmap, offset
