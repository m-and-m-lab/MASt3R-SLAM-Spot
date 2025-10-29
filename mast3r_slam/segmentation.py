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

        # Store 3D points of tracked object
        self.object_point_indices = None
        self.object_3d_points = None  # ← ADD THIS LINE

        if self.enabled:
            try:
                self.model = FastSAM(model_path)
                print(f"✓ FastSAM loaded successfully")
                if self.track_single_object:
                    print(f"✓ Single object tracking enabled (center + 3D projection)")
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
            # Run FastSAM with adjusted parameters
            results = self.model(
                img_uint8,
                device=self.device,
                retina_masks=True,
                imgsz=512,
                conf=0.6,  # ← CHANGED: stricter detection
                iou=0.7,  # ← CHANGED: separate objects better
            )

            if len(results) == 0 or results[0].masks is None:
                print("⚠ FastSAM found no masks")
                return self._handle_lost_object(img.shape[:2])

            masks = results[0].masks.data.cpu().numpy()
            print(f"FastSAM found {len(masks)} masks")  # ← ADD: debug info

            if not self.initialized:
                # First frame: select object at center
                mask = self._select_initial_object(results)
                if mask is not None:
                    self.initialized = True
                    self.frames_since_detection = 0
                    if current_pose is not None:
                        self.last_pose = current_pose

                    # Mark which SLAM points belong to this object
                    if frame is not None:
                        self._mark_object_points(
                            mask, frame, current_pose
                        )  # ← CHANGED: pass pose

                    return self._create_single_color_mask(mask, img.shape[:2])
                return None

            else:
                # Subsequent frames: use tracking
                mask = self._track_with_pose(
                    img_uint8,
                    results,
                    current_pose,
                    img.shape[:2],
                    frame,  # ← CHANGED: pass frame
                )

                if mask is not None:
                    self.frames_since_detection = 0

                    # Update which SLAM points belong to this object
                    if frame is not None:
                        self._mark_object_points(
                            mask, frame, current_pose
                        )  # ← CHANGED: pass pose

                    return self._create_single_color_mask(mask, img.shape[:2])
                else:
                    return self._handle_lost_object(img.shape[:2])

        except Exception as e:
            print(f"Segmentation failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _select_initial_object(self, results):
        """Select object at center of image, excluding very large masks"""
        try:
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) == 0:
                print("✗ No masks found in first frame")
                return None

            H, W = masks[0].shape
            total_pixels = H * W
            center_y, center_x = H // 2, W // 2

            # Filter parameters
            max_size_ratio = 0.4  # Reject masks > 40% of image
            min_size_pixels = 500  # Reject tiny masks

            # Debug: show all masks
            for i, mask in enumerate(masks):
                area = mask.sum()
                area_ratio = area / total_pixels
                print(f"  Mask {i}: {int(area)} pixels ({area_ratio:.1%} of image)")

            # First pass: find mask at center that's not too large
            for i, mask in enumerate(masks):
                area = mask.sum()
                area_ratio = area / total_pixels

                if area_ratio > max_size_ratio:
                    print(f"  → Skipping mask {i} (too large)")
                    continue

                if area < min_size_pixels:
                    print(f"  → Skipping mask {i} (too small)")
                    continue

                if mask[center_y, center_x]:
                    self.object_mask_history.append(mask)
                    print(
                        f"✓ Selected mask {i} at center - {int(area)} pixels ({area_ratio:.1%})"
                    )
                    return mask

            # Second pass: find closest suitable mask
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
                self.object_mask_history.append(best_mask)
                area = best_mask.sum()
                print(
                    f"✓ Selected closest mask {best_mask_idx} - {int(area)} pixels, distance: {min_distance:.1f}"
                )
                return best_mask

            print("✗ Could not find any suitable mask")
            return None

        except Exception as e:
            print(f"Initial object selection failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _track_with_pose(self, img_uint8, results, current_pose, img_shape, frame=None):
        """Track object using 3D projection + IoU fallback"""
        if current_pose is None or self.last_pose is None:
            return self._select_initial_object(results)

        masks = results[0].masks.data.cpu().numpy()

        # PRIMARY: Use 3D projection if available
        if self.object_3d_points is not None and len(self.object_3d_points) > 50:
            predicted_bbox = self._project_3d_to_bbox(
                self.object_3d_points, current_pose, img_shape
            )

            if predicted_bbox is not None:
                x_min, y_min, x_max, y_max = predicted_bbox
                H, W = img_shape[:2]

                best_mask = None
                best_overlap = 0

                bbox_mask = np.zeros((H, W), dtype=bool)
                bbox_mask[y_min:y_max, x_min:x_max] = True

                for mask in masks:
                    overlap = (mask.astype(bool) & bbox_mask).sum()
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_mask = mask

                if best_mask is not None and best_overlap > 100:
                    self.object_mask_history.append(best_mask)
                    self.last_pose = current_pose
                    print(
                        f"✓ Tracked via 3D projection (overlap: {best_overlap} pixels)"
                    )
                    return best_mask

        # FALLBACK: IoU for small movements
        translation = np.linalg.norm(current_pose[:3, 3] - self.last_pose[:3, 3])
        if translation < 0.2 and len(self.object_mask_history) > 0:
            prev_mask = self.object_mask_history[-1]
            best_mask = self._find_best_overlap(masks, prev_mask)
            if best_mask is not None:
                self.object_mask_history.append(best_mask)
                self.last_pose = current_pose
                print(f"✓ Tracked via IoU (camera moved {translation:.3f}m)")
                return best_mask

        # LAST RESORT: Re-select
        print(f"⚠ Lost tracking, re-selecting...")
        mask = self._select_initial_object(results)
        if mask is not None:
            self.last_pose = current_pose
        return mask

    def _project_3d_to_bbox(self, points_3d, current_pose, img_shape):
        """Project 3D object points to 2D bbox"""
        try:
            H, W = img_shape[:2]
            T_CW = np.linalg.inv(current_pose)
            points_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
            points_cam = (T_CW @ points_h.T).T[:, :3]

            valid = points_cam[:, 2] > 0.1
            if valid.sum() < 10:
                return None

            points_cam = points_cam[valid]
            focal_length = max(W, H) * 0.8
            cx, cy = W / 2, H / 2

            x_2d = (points_cam[:, 0] * focal_length / points_cam[:, 2]) + cx
            y_2d = (points_cam[:, 1] * focal_length / points_cam[:, 2]) + cy

            valid_2d = (x_2d >= 0) & (x_2d < W) & (y_2d >= 0) & (y_2d < H)
            if valid_2d.sum() < 10:
                return None

            x_2d = x_2d[valid_2d]
            y_2d = y_2d[valid_2d]

            margin = 30
            bbox = [
                max(0, int(x_2d.min()) - margin),
                max(0, int(y_2d.min()) - margin),
                min(W, int(x_2d.max()) + margin),
                min(H, int(y_2d.max()) + margin),
            ]

            print(f"  Projected {len(x_2d)} points to bbox {bbox}")
            return bbox

        except Exception as e:
            return None

    def _find_best_overlap(self, masks, prev_mask, overlap_threshold=0.3):
        """Find mask with best IoU"""
        best_iou = 0
        best_mask = None

        for mask in masks:
            mask_bool = mask.astype(bool)
            prev_bool = prev_mask.astype(bool)

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

    def _mark_object_points(self, mask, frame, current_pose):
        """Store 3D world coordinates of tracked object"""
        if frame is None or frame.X_canon is None or current_pose is None:
            return

        try:
            h, w = frame.img_shape.flatten().cpu().numpy()
            mask_resized = cv2.resize(mask.astype(np.uint8), (int(w), int(h)))
            mask_flat = mask_resized.flatten().astype(bool)

            self.object_point_indices = mask_flat

            if frame.X_canon.shape[0] == mask_flat.shape[0]:
                # Get 3D points in camera frame
                object_points_cam = frame.X_canon[mask_flat].cpu().numpy()

                # Transform to world frame
                points_h = np.hstack(
                    [object_points_cam, np.ones((object_points_cam.shape[0], 1))]
                )
                points_world = (current_pose @ points_h.T).T[:, :3]

                # Store or accumulate
                if self.object_3d_points is None:
                    self.object_3d_points = points_world
                    print(f"  Initialized 3D model with {len(points_world)} points")
                else:
                    self.object_3d_points = np.vstack(
                        [self.object_3d_points, points_world]
                    )

                    # Downsample if too large
                    if len(self.object_3d_points) > 10000:
                        indices = np.random.choice(
                            len(self.object_3d_points), 10000, replace=False
                        )
                        self.object_3d_points = self.object_3d_points[indices]

                    print(
                        f"  Updated 3D model (total: {len(self.object_3d_points)} points)"
                    )

        except Exception as e:
            print(f"Failed to store object points: {e}")

    def _handle_lost_object(self, img_shape):
        """Handle lost object"""
        self.frames_since_detection += 1

        if self.frames_since_detection > self.max_frames_without_detection:
            print(f"⚠ Lost object, resetting...")
            self.initialized = False
            self.object_mask_history = []
            self.object_point_indices = None
            self.object_3d_points = None  # ← ADD THIS
            return None

        print(
            f"⚠ Object not found ({self.frames_since_detection}/{self.max_frames_without_detection})"
        )
        if len(self.object_mask_history) > 0:
            return self._create_single_color_mask(
                self.object_mask_history[-1], img_shape, alpha=0.5
            )
        return None

    def _create_single_color_mask(self, mask, img_shape, alpha=1.0):
        """Create colored mask"""
        H, W = img_shape[:2]
        colored = np.zeros((H, W, 3), dtype=np.float32)
        mask_bool = mask.astype(bool)
        colored[mask_bool] = self.object_color * alpha
        return colored

    def _segment_everything(self, img):
        """Multi-object segmentation"""
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
        """Assign random colors"""
        H, W = shape
        colored = np.zeros((H, W, 3), dtype=np.float32)

        np.random.seed(42)
        colors = np.random.rand(len(masks), 3)

        for i, mask in enumerate(masks):
            mask_bool = mask.astype(bool)
            colored[mask_bool] = colors[i]

        return colored

    def reset_tracking(self):
        """Reset tracking"""
        self.initialized = False
        self.object_mask_history = []
        self.object_point_indices = None
        self.object_3d_points = None
        self.last_pose = None
        self.frames_since_detection = 0
        print("Tracking reset")
