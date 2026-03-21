"""
MiDaS monocular depth estimation wrapper.

Always uses MiDaS_small (DPT_Large is too slow for real-time assistive use).
Runs on CUDA > MPS > CPU, whichever is available.
Output is normalized per-frame to 0-1 where 1 = NEAR (MiDaS outputs
inverse depth — higher raw value means closer).
"""

from __future__ import annotations
import numpy as np
import torch
import cv2

# Depth estimation does not need high resolution to tell a wall is close.
# Downscaling before the MiDaS transform cuts inference time substantially.
DEPTH_W, DEPTH_H = 256, 192


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DepthEstimator:
    def __init__(self):
        self.device = _pick_device()
        self.model_type = "MiDaS_small"

        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
        self.model.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform

    @torch.inference_mode()
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_bgr: (H, W, 3) uint8 BGR frame (OpenCV convention).
        Returns:
            (H, W) float32 depth map, normalized 0-1, 1 = near.
            Same spatial dimensions as the input frame.
        """
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (DEPTH_W, DEPTH_H),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)

        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = pred.detach().cpu().numpy().astype(np.float32)

        # Per-frame min-max normalize. MiDaS inverse depth: high = close.
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin < 1e-6:
            return np.zeros_like(depth)
        return (depth - dmin) / (dmax - dmin)

    def info(self) -> str:
        return f"MiDaS {self.model_type} on {self.device}"
