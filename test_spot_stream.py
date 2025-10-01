import os
import cv2
import argparse

import numpy as np
from bosdyn.client import create_standard_sdk
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
from mast3r_slam.config import config, load_config


from mast3r_slam.dataloader import Intrinsics, SpotCameraStream


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")

    args = parser.parse_args()
    load_config(args.config)
    print(args.dataset)
    print(config)
    print("Starting Spot camera stream test...")
    try:
        spot = SpotCameraStream()
        stream = spot.frames()

        print(f"Spot Intrinsics: {spot.intrinsics}")

        camera_intrinsics = Intrinsics.from_calib(512, 640, 480, spot.intrinsics, True)
        print(f"Camera Intrinsics: {camera_intrinsics}")
        
        while True:
            print("Getting frame...")
            response = next(stream)
            
            print(f"Received image with format: {response.shot.image.format}")

            # The data is JPEG compressed, so we need to decode it
            data = np.frombuffer(response.shot.image.data, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if img is None:
                print("Failed to decode image. The data might not be a valid JPEG.")
                continue

            print(f"Decoded image shape: {img.shape}")
            
            # Display the image
            cv2.imshow("Spot Camera Stream", img)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Stream test finished.")

if __name__ == "__main__":
    main()
