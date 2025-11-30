# FILE: process.py
#
# DESC:
#   process.py processes given hand keypoint data into different forms to test different training results
#   depending on certain data processing procedures
#
#

import numpy as np
import cv2 as cv


def frame_process(img: np.ndarray, min_pt: tuple[int, int], max_pt: tuple[int, int], keypoints: list[tuple[float, float]], is_left: bool, display_hand_seg: bool = False):

    kp_array = np.asarray(keypoints, dtype=np.float32)
    x_min, y_min = min_pt
    x_max, y_max = max_pt
    H = img.shape[0]
    W = img.shape[1]

    if not is_left:
        kp_array[:, 0] = 1.0 - kp_array[:, 0]
        left_x_min = W - x_max - 1
        left_x_max = W - x_min - 1
        x_min, x_max = left_x_min, left_x_max

    kp_org = kp_array * np.asarray([W, H], dtype=np.float32)

    crop_W = (x_max - x_min)
    crop_H = (y_max - y_min)

    kp_new = kp_org - np.asarray([x_min, y_min], dtype=np.float32)
    kp_new = kp_new / np.asanyarray([crop_W, crop_H], dtype=np.float32)

    kp_new_list = kp_new.tolist()

    print(kp_new)

    if display_hand_seg:
        if not is_left:
            print('         NOT LEFT')
            img = hand_invert(img)

        print(f'{y_min, y_max, x_min, x_max}')
        img = img[y_min:y_max+1, x_min:x_max+1]
        cv.imshow('hand_segment', img)

    return True


def hand_invert(img: np.ndarray):

    return cv.flip(img, 1)   # 1 = horizontal flip


def upright_KP(keypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not isinstance(keypoints, list):
        raise TypeError(
            (f"Process.py: NORMALIZE_SCALE_KP() type not taken {type(keypoints)})"))

    if not len(keypoints) == 21:
        raise IndexError(
            (f"Process.py: NORMALIZE_SCALE_KP() passed in list has length greater or less than 21 (len: {len(keypoints)})"))

    np_keypoints = np.array(keypoints, dtype=np.float32)

    wrist = np_keypoints[0]           # (2,)

    vec = np_keypoints[9]                # (2,)
    vx, vy = vec[0], vec[1]

    angle = np.arctan2(vy, vx)

    target = -np.pi / 2
    theta = target - angle

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    rotated = np_keypoints @ R.T               # (21, 2)
    rotated[np.abs(rotated) < 1e-6] = 0.0
    print(rotated.tolist())

    return rotated.tolist()


def normalize_scale_KP(keypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:

    if not isinstance(keypoints, list):
        raise TypeError(
            (f"Process.py: NORMALIZE_SCALE_KP() type not taken {type(keypoints)})"))

    if not len(keypoints) == 21:
        raise IndexError(
            (f"Process.py: NORMALIZE_SCALE_KP() passed in list has length greater or less than 21 (len: {len(keypoints)})"))

    np_keypoints = np.array(keypoints, dtype=np.float32)
    wrist = np_keypoints[0, :]
    center_wrist_KP = np_keypoints[:, :] - wrist

    dist = np.linalg.norm(center_wrist_KP[9, :])

    if dist < 1e-6:
        return [[999] * 21]

    scaled_CW_KP = center_wrist_KP / dist

    print(scaled_CW_KP.tolist())  # debug

    return scaled_CW_KP.tolist()
