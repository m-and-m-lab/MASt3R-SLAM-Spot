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
    Segmentation module using YOLOv8-seg for pixel-precise segmentation masks.
    Works with 80 COCO classes out of the box.
    """

    def __init__(
        self,
        device="cuda",
        enabled=False,
        object_classes=None,
    ):
        """
        Args:
            device: Device to run model on
            enabled: Whether segmentation is enabled
            object_classes: List of COCO class names to track
                           Examples: "cup", ["cup", "bottle", "book"]
        """
        self.enabled = enabled
        self.model = None
        self.device = device

        # Convert to list
        if isinstance(object_classes, str):
            self.object_classes = [object_classes]
        else:
            self.object_classes = object_classes

        if not self.enabled:
            return

        if YOLO is None:
            print("❌ YOLO not available. Install: pip install ultralytics")
            self.enabled = False
            return

        try:
            print(f"📥 Loading YOLOv8-seg (segmentation model)...")

            # USE -seg MODEL FOR SEGMENTATION MASKS!
            # Options:
            #   yolov8n-seg.pt (nano, 6MB, fastest)
            #   yolov8s-seg.pt (small, 22MB)
            #   yolov8m-seg.pt (medium, 52MB)
            #   yolov8l-seg.pt (large, 83MB)
            self.model = YOLO("yolov8n-seg.pt")  # This gives MASKS!
            self.model.to(device)

            print(f"✓ YOLOv8-seg loaded successfully")

            # Get available class names
            self.class_names = self.model.names

            if self.object_classes:
                # Find class IDs for requested classes
                self.filter_class_ids = []
                for obj_class in self.object_classes:
                    for class_id, class_name in self.class_names.items():
                        if obj_class.lower() in class_name.lower():
                            self.filter_class_ids.append(class_id)

                if self.filter_class_ids:
                    print(
                        f"✓ Tracking: {[self.class_names[i] for i in self.filter_class_ids]}"
                    )
                else:
                    print(f"⚠ No matching classes for {self.object_classes}")
                    print(f"  Available: person, bicycle, car, motorcycle, airplane,")
                    print(f"            bus, train, truck, boat, bottle, cup, fork,")
                    print(f"            knife, spoon, bowl, laptop, mouse, keyboard,")
                    print(f"            cell phone, book, clock, vase, etc.")
                    self.filter_class_ids = None
            else:
                self.filter_class_ids = None
                print(f"✓ Detecting all 80 COCO classes")

        except Exception as e:
            print(f"❌ Failed to load YOLOv8: {e}")
            import traceback

            traceback.print_exc()
            self.enabled = False

    def segment_image(self, img, current_pose=None, frame=None):
        """
        Segment objects using YOLOv8-seg.
        Returns pixel-precise segmentation masks!

        Args:
            img: numpy array (H, W, 3) in RGB, values 0-1 (float)

        Returns:
            List of (mask, class_name, confidence) tuples
            Each mask is (H, W) boolean array - EXACT same as FastSAM!
        """
        if not self.enabled or self.model is None:
            return None

        img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img

        try:
            # Run YOLOv8 segmentation
            results = self.model.predict(
                img_uint8,
                conf=0.25,
                iou=0.5,
                verbose=False,
                classes=self.filter_class_ids,
            )

            if len(results) == 0:
                return None

            result = results[0]

            # THIS IS THE KEY: result.masks contains pixel-precise masks!
            if result.masks is None or len(result.masks) == 0:
                return None

            # Get masks - SAME FORMAT AS FASTSAM!
            masks = result.masks.data.cpu().numpy()  # (N, H, W) boolean masks
            boxes = result.boxes

            # Get class names and confidences
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_names = [self.class_names[cls_id] for cls_id in class_ids]
            confidences = boxes.conf.cpu().numpy()

            # Filter by quality
            H, W = img_uint8.shape[:2]
            total_pixels = H * W

            filtered_results = []
            for mask, class_name, conf in zip(masks, class_names, confidences):
                area_ratio = mask.sum() / total_pixels

                if 0.005 < area_ratio < 0.7 and conf > 0.25:
                    # Return EXACT same format as FastSAM!
                    filtered_results.append((mask, class_name, float(conf)))

            return filtered_results if filtered_results else None

        except Exception as e:
            print(f"YOLOv8 segmentation failed: {e}")
            import traceback

            traceback.print_exc()
            return None
