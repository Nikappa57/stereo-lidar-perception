"""Target generation utilities for BEV object detection training."""

import torch

import globals as G

# Column layout of boxes_3d_ego (BoundingBoxSE3Index, see data.py):
#   0:x 1:y 2:z  3:qw 4:qx 5:qy 6:qz  7:length 8:width 9:height
_X, _Y = 0, 1
_LENGTH, _WIDTH = 7, 8


class TargetEncoder:
    """Generates ground truth heatmaps and offsets from 3D bounding boxes.

    Axis convention (shared with the splat / scatter / decoder): heatmap and
    offset are ``(*, nx, ny)`` with **dim 0 = x** (forward, ``nx``) and
    **dim 1 = y** (lateral, ``ny``); the flat index is ``ix * ny + iy``.
    """

    def __init__(self, num_classes: int = G.NUM_CLASSES):
        self.num_classes = num_classes
        self.x_min = G.X_RANGE[0]
        self.y_min = G.Y_RANGE[0]
        self.res = G.BEV_RES_M
        self.nx = int((G.X_RANGE[1] - G.X_RANGE[0]) / G.BEV_RES_M)
        self.ny = int((G.Y_RANGE[1] - G.Y_RANGE[0]) / G.BEV_RES_M)

    def _draw_gaussian(self, heatmap, center, radius):
        """Draw a 2D Gaussian into ``heatmap`` (shape ``(nx, ny)``) peaked at
        ``center = (ix, iy)``. dim 0 is x (``nx``), dim 1 is y (``ny``)."""
        ix, iy = center
        sigma = radius / 3.0

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

            # 2. radius (cells) from the box footprint (max of length/width)
            radius_m = max(boxes_3d[i, _LENGTH], boxes_3d[i, _WIDTH]) / 2.0
            radius_cells = max(1, int(radius_m / self.res))

            ix, iy = int(gx), int(gy)
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                cls = int(classes[i])
                # 3. Gaussian peak on this class' channel
                self._draw_gaussian(heatmap[cls], (ix, iy), radius_cells)
                # 4. sub-cell offset of the true centre
                offset[0, ix, iy] = gx - ix
                offset[1, ix, iy] = gy - iy

        return heatmap, offset
