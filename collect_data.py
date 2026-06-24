"""
collect_data.py
----------------
Automatic photo capture for training data. Hold each object in front of the
laptop camera and slowly rotate it; the script saves photos by itself into
data/<class_name>/. Repeats for all 6 objects.

(Used to build the dataset for the optional trained-model approach. The final
project uses pure colour+contour vision and does NOT require this — kept for
completeness / the report.)
"""

import cv2
import os
import time

CLASSES = [
    "green_ball",
    "blue_ball",
    "red_ball",
    "blue_cube",
    "purple_cube",
    "yellow_cube",
]

DATA_DIR = "data"
PHOTOS_PER_CLASS = 80
CAPTURE_INTERVAL = 0.5  # seconds between auto captures


def collect_for_class(class_name):
    save_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return
    print(f"\n{'='*50}")
    print(f"  OBJECT: {class_name.upper().replace('_', ' ')}")
    print(f"  Hold the object in front of the camera.")
    print(f"  Slowly rotate it while photos are taken.")
    print(f"  Target: {PHOTOS_PER_CLASS} photos")
    print(f"{'='*50}")
    input("  Press ENTER to start capturing...\n")
    count = 0
    last_capture = time.time()
    print("  Capturing... rotate the object slowly!")
    while count < PHOTOS_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            continue
        now = time.time()
        if now - last_capture >= CAPTURE_INTERVAL:
            filename = os.path.join(save_dir, f"{class_name}_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            last_capture = now
            bar = int(40 * count / PHOTOS_PER_CLASS)
            print(f"  [{'#' * bar}{'.' * (40 - bar)}] {count}/{PHOTOS_PER_CLASS}", end="\r")
    cap.release()
    print(f"\n  Done! Saved {count} photos for '{class_name}'")


def main():
    print("\n" + "=" * 50)
    print("  ROBODOG DATA COLLECTION")
    print(f"  Collecting photos for {len(CLASSES)} objects.")
    print("=" * 50)
    for i, class_name in enumerate(CLASSES):
        print(f"\nObject {i+1} of {len(CLASSES)}: {class_name.upper().replace('_', ' ')}")
        input("  Press ENTER when you have the object ready...")
        collect_for_class(class_name)
    print("\n  ALL DONE! Check your data/ folder.")


if __name__ == "__main__":
    main()
