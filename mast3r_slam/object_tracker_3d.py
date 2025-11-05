import numpy as np
import cv2
import torch
from typing import Dict, List, Optional, Tuple
import dataclasses


@dataclasses.dataclass
class TrackedObject:
    """Represents a tracked object with accumulated 3D points"""

    object_id: int
    class_name: str
    points_world: np.ndarray  # (N, 3) accumulated 3D points
    centroid: np.ndarray  # (3,) current centroid
    color: np.ndarray  # (3,) RGB color for visualization
    first_seen_frame: int
    last_seen_frame: int
    confidence: float = 1.0


class MultiObjectTracker3D:
    """
    Tracks multiple objects across frames using 3D point cloud matching.
    Objects are identified by class name (from YOLO) and tracked via spatial proximity.
    """

    def __init__(
        self,
        segmentation_module,
        match_distance_threshold: float = 0.25,  # 25cm
        max_points_per_object: int = 10000,
        min_points_for_tracking: int = 100,
    ):
        """
        Args:
            segmentation_module: SegmentationModule instance
            match_distance_threshold: Max distance (meters) to consider same object
            max_points_per_object: Downsample if more points accumulated
            min_points_for_tracking: Min points needed to register an object
        """
        self.segmentation = segmentation_module
        self.match_threshold = match_distance_threshold
        self.max_points = max_points_per_object
        self.min_points = min_points_for_tracking

        # Object registry
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0

        # Color palette per class
        np.random.seed(42)
        self.class_colors = {}

    def _get_class_color(self, class_name: str, instance_id: int) -> np.ndarray:
        """Get consistent color for a class with slight variation per instance"""
        if class_name not in self.class_colors:
            self.class_colors[class_name] = np.random.rand(3).astype(np.float32)

        # Use class color with slight variation per instance
        base_color = self.class_colors[class_name]
        variation = np.random.RandomState(instance_id).rand(3) * 0.2 - 0.1
        color = np.clip(base_color + variation, 0, 1)
        return color

    def segment_and_track(
        self,
        img: np.ndarray,
        frame,
        T_WC: np.ndarray,
        frame_id: int,
    ) -> Optional[np.ndarray]:
        """
        Segment objects in current frame and track them in 3D.

        Args:
            img: RGB image (H, W, 3) in 0-1 range
            frame: Frame object with X_canon point cloud
            T_WC: 4x4 camera-to-world transform
            frame_id: Current frame index

        Returns:
            colored_mask: (H, W, 3) segmentation visualization with object IDs
        """
        if not self.segmentation.enabled:
            return None

        # Get segmentation results: [(mask, class_name, confidence), ...]
        results = self.segmentation.segment_image(img, frame=frame)
        if results is None or len(results) == 0:
            print("⚠ No objects detected")
            return None

        print(f"\n🔍 Frame {frame_id}: Found {len(results)} object(s)")

        # Process each detected object
        colored_mask = np.zeros((*img.shape[:2], 3), dtype=np.float32)

        for result_idx, (mask, class_name, confidence) in enumerate(results):
            # Extract 3D points for this mask
            points_world, points_valid = self._extract_3d_points(mask, frame, T_WC)

            if points_world is None or len(points_world) < self.min_points:
                print(
                    f"  ✗ {class_name} {result_idx}: Too few points "
                    f"({len(points_world) if points_world is not None else 0})"
                )
                continue

            centroid = points_world.mean(axis=0)

            # Match to existing object or create new
            obj_id = self._match_or_create(
                centroid, points_world, class_name, confidence, frame_id
            )

            # Update object
            self._update_object(obj_id, points_world, centroid, frame_id)

            # Color the mask
            obj = self.objects[obj_id]
            mask_bool = mask.astype(bool)
            colored_mask[mask_bool] = obj.color

            print(
                f"  ✓ {class_name} → ID={obj_id} "
                f"({len(obj.points_world)} pts, "
                f"centroid=[{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}])"
            )

        return colored_mask

    def _extract_3d_points(
        self, mask: np.ndarray, frame, T_WC: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Extract 3D points in world coordinates for a given mask.

        Returns:
            points_world: (N, 3) or None
            valid: Whether extraction succeeded
        """
        try:
            # Resize mask to match point cloud resolution
            h, w = frame.img_shape.flatten().cpu().numpy()
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (int(w), int(h)), interpolation=cv2.INTER_NEAREST
            )
            mask_flat = mask_resized.flatten().astype(bool)

            # Get 3D points in camera frame
            if frame.X_canon is None:
                return None, False

            points_camera = frame.X_canon.cpu().numpy()  # (H*W, 3)

            # Filter by mask
            masked_points = points_camera[mask_flat]

            if len(masked_points) == 0:
                return None, False

            # Transform to world coordinates
            ones = np.ones((len(masked_points), 1))
            points_homog = np.hstack([masked_points, ones])  # (N, 4)
            points_world_homog = (T_WC @ points_homog.T).T  # (N, 4)
            points_world = points_world_homog[:, :3]  # (N, 3)

            # Filter invalid points
            valid_mask = (
                (points_world[:, 2] > 0.1)  # At least 10cm above ground
                & (points_world[:, 2] < 3.0)  # Below 3m height
                & (np.linalg.norm(points_world, axis=1) < 10.0)  # Within 10m
            )

            points_world = points_world[valid_mask]

            return points_world, True

        except Exception as e:
            print(f"Failed to extract 3D points: {e}")
            import traceback

            traceback.print_exc()
            return None, False

    def _match_or_create(
        self,
        centroid: np.ndarray,
        points: np.ndarray,
        class_name: str,
        confidence: float,
        frame_id: int,
    ) -> int:
        """
        Match centroid to existing object of same class or create new one.

        Returns:
            object_id
        """
        # Only match within same class
        candidates = [
            (obj_id, obj)
            for obj_id, obj in self.objects.items()
            if obj.class_name == class_name
        ]

        best_match = None
        best_distance = float("inf")

        for obj_id, obj in candidates:
            distance = np.linalg.norm(centroid - obj.centroid)

            if distance < self.match_threshold and distance < best_distance:
                best_match = obj_id
                best_distance = distance

        if best_match is not None:
            print(
                f"    → Matched to existing {class_name} ID={best_match} (dist={best_distance:.3f}m)"
            )
            return best_match
        else:
            # Create new object
            new_id = self.next_id
            self.next_id += 1

            self.objects[new_id] = TrackedObject(
                object_id=new_id,
                class_name=class_name,
                points_world=points.copy(),
                centroid=centroid.copy(),
                color=self._get_class_color(class_name, new_id),
                first_seen_frame=frame_id,
                last_seen_frame=frame_id,
                confidence=confidence,
            )

            print(f"    → Created new {class_name} ID={new_id}")
            return new_id

    def _update_object(
        self,
        obj_id: int,
        new_points: np.ndarray,
        new_centroid: np.ndarray,
        frame_id: int,
    ):
        """Accumulate new points to existing object"""
        obj = self.objects[obj_id]

        # Accumulate points
        obj.points_world = np.vstack([obj.points_world, new_points])

        # Downsample if too many points
        if len(obj.points_world) > self.max_points:
            indices = np.random.choice(
                len(obj.points_world), self.max_points, replace=False
            )
            obj.points_world = obj.points_world[indices]

        # Update centroid and metadata
        obj.centroid = obj.points_world.mean(axis=0)
        obj.last_seen_frame = frame_id

    def get_object_by_id(self, obj_id: int) -> Optional[TrackedObject]:
        """Get tracked object by ID"""
        return self.objects.get(obj_id)

    def get_all_objects(self) -> List[TrackedObject]:
        """Get all tracked objects"""
        return list(self.objects.values())

    def save_objects_to_file(self, filepath: str):
        """Save all tracked objects as PLY point clouds"""
        from pathlib import Path

        output_dir = Path(filepath)
        output_dir.mkdir(exist_ok=True, parents=True)

        for obj_id, obj in self.objects.items():
            ply_path = output_dir / f"object_{obj_id}_{obj.class_name}.ply"
            self._save_ply(str(ply_path), obj.points_world, obj.color)
            print(f"💾 Saved object {obj_id} ({obj.class_name}) to {ply_path}")

    def _save_ply(self, filepath: str, points: np.ndarray, color: np.ndarray):
        """Save point cloud as PLY file"""
        with open(filepath, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color_uint8 = (color * 255).astype(np.uint8)
            for point in points:
                f.write(
                    f"{point[0]} {point[1]} {point[2]} "
                    f"{color_uint8[0]} {color_uint8[1]} {color_uint8[2]}\n"
                )
