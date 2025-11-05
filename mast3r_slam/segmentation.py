import sys
from pathlib import Path
import numpy as np
import torch
import cv2

try:
    from fastsam import FastSAM, FastSAMPrompt
except ImportError:
    FastSAM = None
    FastSAMPrompt = None
    print("Warning: FastSAM not found. Segmentation disabled.")


class SegmentationModule:
    def __init__(
        self,
        model_path="FastSAM-x.pt",
        device="cuda",
        enabled=False,
        object_name=None,
        track_single_object=True,
    ):
        self.enabled = enabled and FastSAM is not None
        self.model = None
        self.device = device
        self.track_single_object = track_single_object
        self.object_name = object_name

        # Simple per-frame coloring (no tracking across frames)
        self.object_color = np.array([1.0, 0.0, 0.0])  # Red for single object mode

        if self.enabled:
            try:
                self.model = FastSAM(model_path)
                print(f"✓ FastSAM loaded successfully")
                if self.track_single_object:
                    print(f"✓ Single object mode: segments center object per keyframe")
                else:
                    print(f"✓ Multi-object mode: segments all objects per keyframe")
            except Exception as e:
                print(f"Failed to load FastSAM model: {e}")
                self.enabled = False

    def segment_image(self, img, current_pose=None, frame=None):
        """
        Segment current frame independently.

        Args:
            img: numpy array (H, W, 3) in RGB, values 0-1 (float)
            current_pose: 4x4 camera pose matrix (unused, kept for compatibility)
            frame: Frame object with X_canon for point cloud coloring

        Returns:
            colored segmentation map (H, W, 3) in 0-1 range
        """
        if not self.enabled or self.model is None:
            return None

        if self.track_single_object:
            return self._segment_single_object(img, frame)
        else:
            return self._segment_everything(img)

    def _segment_single_object(self, img, frame=None):
        """
        Segment object at center of current frame.
        Each keyframe is segmented independently.
        """
        img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img

        try:
            # Run FastSAM
            results = self.model(
                img_uint8,
                device=self.device,
                retina_masks=True,
                imgsz=512,
                conf=0.6,
                iou=0.5,
            )

            if len(results) == 0 or results[0].masks is None:
                print("⚠ FastSAM found no masks")
                return None

            masks = results[0].masks.data.cpu().numpy()
            print(f"FastSAM found {len(masks)} masks")

            # Select object at center (or closest to center)
            mask = self._select_center_object(masks, img.shape[:2])

            if mask is not None:
                # Mark which points in the frame's point cloud belong to this object
                if frame is not None:
                    self._mark_frame_object_points(mask, frame)

                return self._create_single_color_mask(mask, img.shape[:2])

            return None

        except Exception as e:
            print(f"Segmentation failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _select_center_object(self, masks, img_shape):
        """
        Select object at center of image, excluding very large masks.
        This is called independently for each keyframe.
        """
        try:
            if len(masks) == 0:
                print("✗ No masks found")
                return None

            H, W = img_shape[:2]
            total_pixels = H * W
            center_y, center_x = H // 2, W // 2

            # Filter parameters
            max_size_ratio = 0.4  # Reject masks > 40% of image
            min_size_pixels = 500  # Reject tiny masks

            # First pass: find mask at center that's not too large
            for i, mask in enumerate(masks):
                area = mask.sum()
                area_ratio = area / total_pixels

                if area_ratio > max_size_ratio:
                    continue

                if area < min_size_pixels:
                    continue

                if mask[center_y, center_x]:
                    print(
                        f"✓ Selected mask {i} at center - {int(area)} pixels ({area_ratio:.1%})"
                    )
                    return mask

            # Second pass: find closest suitable mask to center
            print("⚠ No suitable mask at center, finding closest...")
            best_mask = None
            best_mask_idx = None
            min_distance = float("inf")

            for i, mask in enumerate(masks):
                area = mask.sum()
                area_ratio = area / total_pixels

                if area_ratio > max_size_ratio or area < min_size_pixels:
                    continue

                ys, xs = np.where(mask)
                if len(ys) > 0:
                    mask_center_y = ys.mean()
                    mask_center_x = xs.mean()
                    distance = np.sqrt(
                        (mask_center_x - center_x) ** 2
                        + (mask_center_y - center_y) ** 2
                    )

                    if distance < min_distance:
                        min_distance = distance
                        best_mask = mask
                        best_mask_idx = i

            if best_mask is not None:
                area = best_mask.sum()
                print(
                    f"✓ Selected closest mask {best_mask_idx} - {int(area)} pixels, distance: {min_distance:.1f}"
                )
                return best_mask

            print("✗ Could not find any suitable mask")
            return None

        except Exception as e:
            print(f"Object selection failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _mark_frame_object_points(self, mask, frame):
        """
        Mark which 3D points in THIS frame's point cloud belong to the segmented object.
        This is done per-keyframe, not accumulated across frames.
        """
        if frame is None or frame.X_canon is None:
            return

        try:
            h, w = frame.img_shape.flatten().cpu().numpy()
            mask_resized = cv2.resize(mask.astype(np.uint8), (int(w), int(h)))
            mask_flat = mask_resized.flatten().astype(bool)

            # Store the boolean mask indicating which points are part of the object
            # This is stored in the frame itself, not accumulated globally
            if hasattr(frame, "X_canon") and frame.X_canon is not None:
                if frame.X_canon.shape[0] == mask_flat.shape[0]:
                    # You can store this as an attribute on the frame
                    frame.object_point_mask = mask_flat
                    num_object_points = mask_flat.sum()
                    print(
                        f"  Marked {num_object_points} points in this keyframe as object"
                    )

        except Exception as e:
            print(f"Failed to mark frame object points: {e}")

    def _create_single_color_mask(self, mask, img_shape, alpha=1.0):
        """Create colored mask for visualization"""
        H, W = img_shape[:2]
        colored = np.zeros((H, W, 3), dtype=np.float32)
        mask_bool = mask.astype(bool)
        colored[mask_bool] = self.object_color * alpha
        return colored

    def _segment_everything(self, img):
        """
        Multi-object segmentation - segments all objects in current frame.
        Each keyframe is segmented independently.
        """
        img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img

        try:
            results = self.model(
                img_uint8,
                device=self.device,
                retina_masks=True,
                imgsz=512,
                conf=0.4,
                iou=0.9,
            )

            if len(results) == 0 or results[0].masks is None:
                return None

            masks = results[0].masks.data.cpu().numpy()
            colored_mask = self._create_multi_colored_mask(masks, img.shape[:2])
            return colored_mask

        except Exception as e:
            print(f"Segmentation failed: {e}")
            return None

    def _create_multi_colored_mask(self, masks, shape):
        """Assign different colors to different objects"""
        H, W = shape
        colored = np.zeros((H, W, 3), dtype=np.float32)

        np.random.seed(42)  # Consistent colors across runs
        colors = np.random.rand(len(masks), 3)

        for i, mask in enumerate(masks):
            mask_bool = mask.astype(bool)
            colored[mask_bool] = colors[i]

        return colored
