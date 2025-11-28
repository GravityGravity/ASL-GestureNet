# File: dataset_ann.py
#
# DESC: Grabs a dataset folder location and runs each image through mediapipe hand keypoint analysis, and writes all keypoints into a ASL_ann.csv file


import os
import sys
import random
from pathlib import Path

import pandas as pd
import mediapipe as mp
import cv2 as cv

print()
csv_fieldsnames = ['image_id', 'image_h', 'image_w', 'label', 'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
                   'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

data_dir = Path.cwd() / 'datasets' / 'train' / 'asl_dataset'  # Dataset Directory
if not data_dir.exists():
    raise FileNotFoundError(f"{data_dir} does not exist")
print(data_dir)  # debug

# Load media pipe hand tracking solution
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

unreadable_imgs = 0

ann_csv = pd.DataFrame(columns=csv_fieldsnames)


def loop_dir(iter_print: int = None):

    print('    loop_dir()')  # debug

    img_count = 0
    jpeg_count = 0
    jpg_count = 0
    png_count = 0

    i_init = random.randint(0, 9)

    for i, img_path in enumerate(data_dir.rglob('*.jpeg', case_sensitive=False)):
        print(img_path.name)
        if not ((i+i_init) % iter_print):
            keypoint_extract(img_path, display_hand=True)
        else:
            keypoint_extract(img_path)
        jpeg_count += 1
        img_count += 1

    print(f'\njpeg_count: {jpeg_count}')  # debug

    for img_path in data_dir.rglob('*.jpg', case_sensitive=False):
        print(img_path)
        jpg_count += 1
        img_count += 1

    print(f'\njpg_count: {jpg_count}')  # debug

    for img_path in data_dir.rglob('*.png', case_sensitive=False):
        print(img_path.name)
        png_count += 1
        img_count += 1

    print(f'\nimg_count: {jpg_count}')  # debug

    print(f'\n  -> undetect_images: {unreadable_imgs}')  # debug
    print(f'\n  -> total_img_count: {img_count}')  # debug

    ann_csv.to_csv("asl_dataset.csv", index=False)

    return True


def write_csv(annotations: list[str]):
    global ann_csv
    ann_csv.loc[len(ann_csv)] = annotations
    return False


def keypoint_extract(img_path: Path, normalize: bool = True, display_hand: bool = False) -> list:

    global unreadable_imgs
    img_name = img_path.name
    img = cv.imread(str(img_path))  # H W C dims
    label = img_path.parent.name
    H = img.shape[0]
    W = img.shape[1]

    img_info = [img_name, H, W, label]
    RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # H, W, C
    result = hand.process(RGB_img)

    if result.multi_hand_landmarks:
        singlehand = result.multi_hand_landmarks[0]

        for lm in singlehand.landmark:
            # print(f' {lm.z}') # debug
            cords = (lm.x, lm.y)
            img_info.append(cords)

        if display_hand:
            mp_drawing.draw_landmarks(
                img, singlehand, mp_hands.HAND_CONNECTIONS)

            cv.imshow(f'hand_class {label}', img)
            key = cv.waitKey(250) & 0xFF

            if key == ord('q'):
                sys.exit(1)
        write_csv(img_info)

    else:

        if display_hand:
            cv.imshow(f'hand_class {label}', img)
            key = cv.waitKey(350) & 0xFF

            if key == ord('q'):
                sys.exit(1)

        unreadable_imgs += 1

    if not normalize:
        pass
    return True


def KP_process():
    pass


loop_dir(iter_print=15)
