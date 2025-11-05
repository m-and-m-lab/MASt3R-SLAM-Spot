import numpy as np
import torch
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("❌ YOLO not found. Install: pip install ultralytics")


class SegmentationModule:
    """
    Segmentation module using YOLO-World for open-vocabulary object detection.
    Supports text prompts like "cup", "book", "laptop", etc.
    """

    def __init__(
        self,
        device="cuda",
        enabled=False,
        object_classes=None,  # List of classes or single class: ["cup", "book"] or "cup"
    ):
        """
        Args:
            device: Device to run model on
            enabled: Whether segmentation is enabled
            object_classes: Text prompt(s) for objects to track
                           Examples: "cup", ["cup", "book", "bottle"]
        """
        self.enabled = enabled
        self.model = None
        self.device = device
        self.object_classes = object_classes

        if not self.enabled:
            return

        if YOLO is None:
            print("❌ YOLO not available. Install: pip install ultralytics")
            self.enabled = False
            return

        try:
            # Load YOLO-World model
            # Options: yolov8s-worldv2.pt (small, fast)
            #          yolov8m-worldv2.pt (medium)
            #          yolov8l-worldv2.pt (large, more accurate)
            self.model = YOLO("yolov8s-worldv2.pt")
            self.model.to(device)

            # Set classes to detect based on text prompts
            if self.object_classes:
                classes = (
                    [self.object_classes]
                    if isinstance(self.object_classes, str)
                    else self.object_classes
                )
                self.model.set_classes(classes)
                print(f"✓ YOLO-World loaded - tracking: {classes}")
            else:
                print(
                    f"✓ YOLO-World loaded - open vocabulary mode (detects common objects)"
                )

        except Exception as e:
            print(f"❌ Failed to load YOLO-World: {e}")
            import traceback

            traceback.print_exc()
            self.enabled = False

    def segment_image(self, img, current_pose=None, frame=None):
        """
        Segment objects using YOLO-World with text prompts.

        Args:
            img: numpy array (H, W, 3) in RGB, values 0-1 (float)
            current_pose: unused (kept for compatibility)
            frame: unused (kept for compatibility)

        Returns:
            List of (mask, class_name, confidence) tuples
            or None if no objects detected
        """
        if not self.enabled or self.model is None:
            return None

        img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img

        try:
            # Run YOLO-World inference
            results = self.model.predict(
                img_uint8,
                conf=0.1,  # Detection confidence threshold
                iou=0.5,  # NMS IoU threshold
                verbose=False,
            )

            if len(results) == 0 or results[0].masks is None:
                return None

            result = results[0]

            # Check if masks exist
            if result.masks is None or len(result.masks) == 0:
                return None

            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            boxes = result.boxes

            # Get class names and confidences
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_names = [result.names[cls_id] for cls_id in class_ids]
            confidences = boxes.conf.cpu().numpy()

            # Filter detections by quality
            H, W = img_uint8.shape[:2]
            total_pixels = H * W

            filtered_results = []
            for mask, class_name, conf in zip(masks, class_names, confidences):
                area_ratio = mask.sum() / total_pixels

                # Keep detections with reasonable size and good confidence
                if 0.005 < area_ratio < 0.7 and conf > 0.15:
                    filtered_results.append((mask, class_name, conf))

            return filtered_results if filtered_results else None

        except Exception as e:
            print(f"YOLO segmentation failed: {e}")
            import traceback

            traceback.print_exc()
            return None
