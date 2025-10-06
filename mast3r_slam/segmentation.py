import sys
from pathlib import Path
import numpy as np
import torch
import cv2

# Add FastSAM to path if needed
fastsam_path = Path(__file__).parent.parent / "thirdparty" / "FastSAM"
if fastsam_path.exists():
    sys.path.insert(0, str(fastsam_path))

try:
    from fastsam import FastSAM, FastSAMPrompt
except ImportError:
    FastSAM = None
    FastSAMPrompt = None
    print("Warning: FastSAM not found. Segmentation disabled.")


class SegmentationModule:
    def __init__(self, model_path="FastSAM-x.pt", device="cuda", enabled=False):
        self.enabled = enabled and FastSAM is not None
        self.model = None
        self.device = device

        if self.enabled:
            try:
                self.model = FastSAM(model_path)
            except Exception as e:
                print(f"Failed to load FastSAM model: {e}")
                self.enabled = False

    def segment_image(self, img):
        """
        img: numpy array (H, W, 3) in RGB, values 0-1 (float)
        Returns: colored segmentation map (H, W, 3) in 0-1 range
        """
        if not self.enabled:
            return None

        # FastSAM expects uint8
        img_uint8 = (img * 255).astype(np.uint8)

        try:
            results = self.model(
                img_uint8,
                device=self.device,
                retina_masks=True,
                imgsz=512,
                conf=0.4,
                iou=0.9,
            )

            prompt_process = FastSAMPrompt(img_uint8, results, device=self.device)
            masks = prompt_process.everything_prompt()

            if masks is None or len(masks) == 0:
                return None

            colored_mask = self.create_colored_mask(masks, img.shape[:2])
            return colored_mask
        except Exception as e:
            print(f"Segmentation failed: {e}")
            return None

    def create_colored_mask(self, masks, shape):
        """Assign random colors to each segment"""
        H, W = shape
        colored = np.zeros((H, W, 3), dtype=np.float32)

        # Generate consistent colors
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3)

        for i, mask in enumerate(masks):
            colored[mask] = colors[i]

        return colored
