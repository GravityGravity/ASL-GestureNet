# FILE: dataset_ann.py
#
# DESC:
#   Build ASL hand-keypoint annotations from an keypoint training dataset.
#   Walks the dataset, runs MediaPipe Hands on each image, and
#   writes normalized keypoints into asl_dataset.csv.
#
#   POTENTIAL FEATURE LIST:
#     - Add keypoint error checking for badly detected hands
#     - Add Z-coordinate handling for hand keypoint analysis
#

import os
import sys
import random
from pathlib import Path

import pandas as pd
import mediapipe as mp
import cv2 as cv

from process import normalize_scale_KP, upright_KP

dataset_type = input('  >Is this for \'train\' or \'test\'?\n   >')
dataset_folder = input('  >What is dataset folder name?\n   >')
csv_name = input('  >What csv name would you like?\n   >')

print()
csv_fieldnames = [
    "image_id",
    "label",
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

# Dataset directory
data_dir = Path.cwd() / "datasets" / dataset_type / dataset_folder
if not data_dir.exists():
    raise FileNotFoundError(f"{data_dir} does not exist")
print(data_dir)  # debug

# MediaPipe Hands configuration
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
)

unreadable_imgs = 0

# Annotation DataFrame
ann_csv = pd.DataFrame(columns=csv_fieldnames)


def loop_dir(iter_print: int) -> bool:
    """Walk the dataset directory and process all image files."""
    print("    loop_dir()")  # debug

    img_count = 0
    jpeg_count = 0
    jpg_count = 0
    png_count = 0

    i_init = random.randint(0, 9)

    # Process .jpeg files
    for i, img_path in enumerate(data_dir.rglob("*.jpeg", case_sensitive=False)):
        print(img_path.name)
        show = not ((i + i_init) % iter_print)
        keypoint_extract(
            img_path,
            center_scale=True,
            normal_angle=True,
            display_hand=show,
        )
        jpeg_count += 1
        img_count += 1

    print(f"\njpeg_count: {jpeg_count}")  # debug

    # Process .jpg files
    for i, img_path in enumerate(data_dir.rglob("*.jpg", case_sensitive=False)):
        print(img_path.name)
        show = not ((i + i_init) % iter_print)
        keypoint_extract(
            img_path,
            center_scale=True,
            normal_angle=True,
            display_hand=show,
        )
        jpg_count += 1
        img_count += 1

    print(f"\njpg_count: {jpg_count}")  # debug

    # Process .png files
    for i, img_path in enumerate(data_dir.rglob("*.png", case_sensitive=False)):
        print(img_path.name)
        show = not ((i + i_init) % iter_print)
        keypoint_extract(
            img_path,
            center_scale=True,
            normal_angle=True,
            display_hand=show,
        )
        png_count += 1
        img_count += 1

    print(f"\npng_count: {png_count}")  # debug
    print(f"\nimg_count: {img_count}")  # debug
    print(f"\n  -> undetect_images: {unreadable_imgs}")  # debug
    print(f"\n  -> total_img_count: {img_count}")  # debug

    # Write annotations to CSV
    ann_csv.to_csv(csv_name, index=False)

    return True


def append_csv(annotations: list) -> None:
    """Append a new annotation row to the CSV DataFrame."""
    global ann_csv
    ann_csv.loc[len(ann_csv)] = annotations


def keypoint_extract(
    img_path: Path,
    center_scale: bool = False,
    normal_angle: bool = False,
    display_hand: bool = False,
) -> bool:
    """Run MediaPipe Hands on an image and store keypoints."""
    global unreadable_imgs

    img_name = img_path.name
    img = cv.imread(str(img_path))  # H, W, C

    if img is None:
        print(f"Warning: could not read image {img_path}")  # debug
        unreadable_imgs += 1
        return False

    label = img_path.parent.name

    img_info = [img_name, label]

    RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = hand.process(RGB_img)

    if result.multi_hand_landmarks:
        singlehand = result.multi_hand_landmarks[0]

        # Collect (x, y) normalized keypoints
        for lm in singlehand.landmark:
            cords = (lm.x, lm.y)
            img_info.append(cords)

        # Center and scale relative to wrist / middle MCP
        if center_scale:
            img_info[2:23] = normalize_scale_KP(img_info[2:23])
            # Sentinel check: first keypoint is [999.0, 999.0] on failure
            if img_info[2][0] == 999.0:
                print('   \n\n FOUND BAD DATASET!!!!!!!!')
                return False

        # Rotate so middle finger is upright
        if normal_angle:
            img_info[2:23] = upright_KP(img_info[2:23])

        # Optional visualization
        if display_hand:
            mp_drawing.draw_landmarks(
                img,
                singlehand,
                mp_hands.HAND_CONNECTIONS,
            )
            cv.imshow(f"hand_class {label}", img)
            key = cv.waitKey(250) & 0xFF
            if key == ord("q"):
                sys.exit(1)

        append_csv(img_info)

    else:
        # No hand detected
        if display_hand:
            cv.imshow(f"hand_class {label}", img)
            key = cv.waitKey(350) & 0xFF
            if key == ord("q"):
                sys.exit(1)
        unreadable_imgs += 1

    return True


if __name__ == "__main__":
    loop_dir(iter_print=15)
