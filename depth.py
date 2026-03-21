"""
MiDaS monocular depth estimation wrapper.

Loads DPT_Large on CUDA if available, otherwise MiDaS_small on CPU.
Output is normalized per-frame to 0-1 where 1 = NEAR (MiDaS outputs
inverse depth — higher raw value means closer).
"""

from __future__ import annotations
import numpy as np
import torch
import cv2


class DepthEstimator:
    def __init__(self, force_small: bool = False):
        self.device = "cuda" if torch.cuda.is_available() and not force_small else "cpu"
        model_type = "DPT_Large" if self.device == "cuda" and not force_small else "MiDaS_small"
        self.model_type = model_type

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "MiDaS_small":
            self.transform = transforms.small_transform
        else:
            self.transform = transforms.dpt_transform

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
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)

        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = pred.cpu().numpy().astype(np.float32)

        # Per-frame min-max normalize. MiDaS inverse depth: high = close.
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin < 1e-6:
            return np.zeros_like(depth)
        return (depth - dmin) / (dmax - dmin)

    def info(self) -> str:
        return f"MiDaS {self.model_type} on {self.device}"
