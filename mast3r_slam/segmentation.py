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

        # Tracking state
        self.object_mask_history = []
        self.object_color = np.array([1.0, 0.0, 0.0])  # Red - consistent color
        self.last_pose = None
        self.initialized = False
        self.frames_since_detection = 0
        self.max_frames_without_detection = 5

        # Store which points in SLAM point cloud belong to tracked object
        self.object_point_indices = None

        # NO CLIP - using IoU-only tracking
        self.clip_model = None
        self.clip_preprocess = None

        if self.enabled:
            try:
                self.model = FastSAM(model_path)
                print(f"✓ FastSAM loaded successfully")
                if self.track_single_object:
                    print(f"✓ Single object tracking enabled (IoU-based, no CLIP)")
            except Exception as e:
                print(f"Failed to load FastSAM model: {e}")
                self.enabled = False

    def segment_image(self, img, current_pose=None, frame=None):
        """
        Main segmentation function

        Args:
            img: numpy array (H, W, 3) in RGB, values 0-1 (float)
            current_pose: 4x4 camera pose matrix (optional)
            frame: Frame object with X_canon, C, etc. (optional)

        Returns:
            colored segmentation map (H, W, 3) in 0-1 range
        """
        if not self.enabled or self.model is None:
            return None

        if self.track_single_object:
            return self._segment_single_object(img, current_pose, frame)
        else:
            return self._segment_everything(img)

    def _segment_single_object(self, img, current_pose=None, frame=None):
        """Track a single object across frames"""
        img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img

        try:
            # Run FastSAM
            results = self.model(
                img_uint8,
                device=self.device,
                retina_masks=True,
                imgsz=512,
                conf=0.4,
                iou=0.9,
            )

            if len(results) == 0 or results[0].masks is None:
                return self._handle_lost_object(img.shape[:2])

            # Strategy based on initialization and camera motion
            if not self.initialized:
                # First frame: select largest mask
                mask = self._select_initial_object(results)
                if mask is not None:
                    self.initialized = True
                    self.frames_since_detection = 0
                    if current_pose is not None:
                        self.last_pose = current_pose

                    # Mark which SLAM points belong to this object
                    if frame is not None:
                        self._mark_object_points(mask, frame)

                    return self._create_single_color_mask(mask, img.shape[:2])
                return None

            else:
                # Subsequent frames: use tracking
                mask = self._track_with_pose(
                    img_uint8, results, current_pose, img.shape[:2]
                )

                if mask is not None:
                    self.frames_since_detection = 0

                    # Update which SLAM points belong to this object
                    if frame is not None:
                        self._mark_object_points(mask, frame)

                    return self._create_single_color_mask(mask, img.shape[:2])
                else:
                    return self._handle_lost_object(img.shape[:2])

        except Exception as e:
            print(f"Segmentation failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _select_initial_object(self, results):
        """Select largest mask as the target object"""
        try:
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) == 0:
                print("✗ No masks found in first frame")
                return None

            # Find largest mask (most pixels)
            areas = [mask.sum() for mask in masks]
            largest_idx = np.argmax(areas)
            mask = masks[largest_idx]

            self.object_mask_history.append(mask)
            print(
                f"✓ Selected largest mask as target ({int(areas[largest_idx])} pixels)"
            )
            return mask
        except Exception as e:
            print(f"Initial object selection failed: {e}")
            return None

    def _track_with_pose(self, img_uint8, results, current_pose, img_shape):
        """Track object using camera pose and previous mask"""
        if current_pose is None or self.last_pose is None:
            # No pose info - fallback to selecting largest mask
            return self._select_initial_object(results)

        # Calculate camera movement
        translation = np.linalg.norm(current_pose[:3, 3] - self.last_pose[:3, 3])

        # Get all masks
        masks = results[0].masks.data.cpu().numpy()

        # Strategy based on camera motion
        if translation < 0.3:  # Small/medium movement - use IoU matching
            if len(self.object_mask_history) > 0:
                prev_mask = self.object_mask_history[-1]
                best_mask = self._find_best_overlap(masks, prev_mask)
                if best_mask is not None:
                    self.object_mask_history.append(best_mask)
                    self.last_pose = current_pose
                    print(f"✓ Tracked via IoU (camera moved {translation:.3f}m)")
                    return best_mask

        # Large movement or no good overlap - re-select largest mask
        print(f"Camera moved {translation:.3f}m, re-selecting object...")
        mask = self._select_initial_object(results)
        if mask is not None:
            self.last_pose = current_pose
        return mask

    def _find_best_overlap(self, masks, prev_mask, overlap_threshold=0.3):
        """Find mask with best IoU overlap with previous mask"""
        best_iou = 0
        best_mask = None

        for mask in masks:
            mask_bool = mask.astype(bool)
            prev_bool = prev_mask.astype(bool)

            # Calculate IoU (Intersection over Union)
            intersection = np.logical_and(mask_bool, prev_bool).sum()
            union = np.logical_or(mask_bool, prev_bool).sum()

            if union > 0:
                iou = intersection / union
                if iou > best_iou and iou > overlap_threshold:
                    best_iou = iou
                    best_mask = mask

        if best_mask is not None:
            print(f"  Best IoU: {best_iou:.3f}")

        return best_mask

    def _mark_object_points(self, mask, frame):
        """Mark which points in SLAM's point cloud belong to tracked object"""
        if frame is None or frame.X_canon is None:
            return

        try:
            # Resize mask to match frame's point cloud shape
            h, w = frame.img_shape.flatten().cpu().numpy()
            mask_resized = cv2.resize(mask.astype(np.uint8), (int(w), int(h)))

            # Flatten to match X_canon shape (H*W, 3)
            mask_flat = mask_resized.flatten().astype(bool)

            # Store indices of points belonging to this object
            self.object_point_indices = mask_flat

            # Optional: Get actual 3D points for future use
            if frame.X_canon.shape[0] == mask_flat.shape[0]:
                object_points_3d = frame.X_canon[mask_flat]
                print(
                    f"  Marked {mask_flat.sum()} / {mask_flat.shape[0]} points as object"
                )

        except Exception as e:
            print(f"Failed to mark object points: {e}")

    def _handle_lost_object(self, img_shape):
        """Handle case when object is temporarily lost"""
        self.frames_since_detection += 1

        if self.frames_since_detection > self.max_frames_without_detection:
            print(
                f"⚠ Lost object for {self.frames_since_detection} frames, resetting..."
            )
            self.initialized = False
            self.object_mask_history = []
            self.object_point_indices = None
            return None

        # Return last known mask with reduced opacity
        print(
            f"⚠ Object not found (frame {self.frames_since_detection}/{self.max_frames_without_detection})"
        )
        if len(self.object_mask_history) > 0:
            return self._create_single_color_mask(
                self.object_mask_history[-1], img_shape, alpha=0.5
            )
        return None

    def _create_single_color_mask(self, mask, img_shape, alpha=1.0):
        """Create colored mask with single consistent color"""
        H, W = img_shape[:2]
        colored = np.zeros((H, W, 3), dtype=np.float32)

        mask_bool = mask.astype(bool)
        colored[mask_bool] = self.object_color * alpha

        return colored

    def _segment_everything(self, img):
        """Original multi-object segmentation"""
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
            colored_mask = self.create_colored_mask(masks, img.shape[:2])
            return colored_mask

        except Exception as e:
            print(f"Segmentation failed: {e}")
            return None

    def create_colored_mask(self, masks, shape):
        """Assign random colors to each segment (multi-object mode)"""
        H, W = shape
        colored = np.zeros((H, W, 3), dtype=np.float32)

        np.random.seed(42)
        colors = np.random.rand(len(masks), 3)

        for i, mask in enumerate(masks):
            mask_bool = mask.astype(bool)
            colored[mask_bool] = colors[i]

        return colored

    def reset_tracking(self):
        """Reset tracking state"""
        self.initialized = False
        self.object_mask_history = []
        self.object_point_indices = None
        self.last_pose = None
        self.frames_since_detection = 0
        print("Tracking reset")
