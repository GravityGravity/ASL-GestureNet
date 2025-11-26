# FILE: Caption.py
#
# DESC: Gets the most confident word from the confidence matrix and sets the word caption underneath the screen

import cv2 as cv
import numpy as np
# Text that appears in the bottom of the video
caption: list[str] = ['CAPTION', 'TEST']
# Text that appears above hand
title: str = 'HAND TEST'

# Caption Related Functions


def write_cap(img: np.ndarray, img_W: int, img_H: int):
    global caption
    if caption:

        # Caption Text openCV parameters
        complete_cap = concate_caption()
        font_size = 0.8
        font_thick = 1
        font_face = cv.FONT_HERSHEY_SIMPLEX
        Wcolor = (255, 255, 255)  # White
        Gcolor = (55, 55, 55)  # Gray

        # Caption Size
        (text_W, text_H), baseline = cv.getTextSize(
            complete_cap, font_face, font_size, font_thick)

        # Caption Position
        x_rel_center = int((img_W - 1) // 2) - int(text_W // 2)
        y_rel = int((img_H - 1) * 0.9)

        # Caption Background (CB)
        CB_min, CB_max = box_pad(
            ((x_rel_center), (y_rel - text_H - baseline)), ((x_rel_center + text_W), (y_rel + baseline)), 1.05)

        img = cv.rectangle(img, CB_min, CB_max, Gcolor, -1)

        return cv.putText(img, complete_cap, (x_rel_center, y_rel), font_face, font_size, Wcolor, font_thick, cv.LINE_AA, bottomLeftOrigin=False)
    else:
        return img


def append_cap(s: str):
    caption.append(s)
    return True


def concate_caption() -> str:
    global caption
    cap = ' '.join(s for s in caption)
    return cap


def clear_cap():
    global caption
    caption.clear()
    caption = []
    return True

# Title Related Functions


def write_title(img: np.ndarray, TL_rect_pt: tuple[int, int], color: tuple[int, int, int]):
    global title
    if title:
        x_cord, y_cord = TL_rect_pt
        # Title Text openCV parameters
        font_size = 0.75
        font_thick = 2
        font_face = cv.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)  # white
        margin = 20

        # Title Size
        (text_W, text_H), baseline = cv.getTextSize(
            title, font_face, font_size, font_thick)

        # Title Background (TB) Position
        TB_min = (x_cord, (y_cord - text_H - baseline * 2))
        TB_max = ((x_cord + text_W + (margin * 2)), (y_cord))

        # Draw title background
        img = cv.rectangle(img, TB_min, TB_max, color, -1)

        return cv.putText(img, title, (x_cord + margin, y_cord - baseline), font_face, font_size, font_color, font_thick, cv.LINE_AA, bottomLeftOrigin=False)
    else:
        return img


def set_title():
    global title
    pass

# Adds padding to bounding boxes (Hand Bbox, Caption Bbox)


def box_pad(pt_min: tuple[int, int], pt_max: tuple[int, int], scale: float = 1.1) -> tuple[tuple, tuple]:
    """
    Add padding to hand bounding box based off of scale.
        Used for:
            - Hand Bounding Box
            - Caption Box
    """
    x_min, y_min = pt_min
    x_max, y_max = pt_max

    box_W = (x_max - x_min)
    box_H = (y_max - y_min)
    pad_W = ((box_W * scale) - box_W) // 2
    pad_H = ((box_H * scale) - box_H) // 2

    new_x_min = int(x_min - pad_W)
    new_y_min = int(y_min - pad_H)
    new_x_max = int(x_max + pad_W)
    new_y_max = int(y_max + pad_H)

    return ((new_x_min, new_y_min), (new_x_max, new_y_max))


def check_OOB(pt: tuple[int, int], H: int, W: int):
    """
    Check if X,Y cords are outside of bounds W x H
    """

    x_cord, y_cord = pt

    if (x_cord > (W - 1)):
        x_cord = (W - 1)
    if (x_cord < 0):
        x_cord = 0

    if (y_cord > H):
        y_cord = (H - 1)
        pass
    if (y_cord < 0):
        y_cord = 0
        pass

    return x_cord, y_cord
