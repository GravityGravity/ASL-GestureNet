# FILE: caption.py
#
# DESC:
#   Draw caption text at the bottom of the frame and a title above the hand
#   bounding box. Also provides small helpers for padding boxes and bounds
#   checking for coordinates.
#

import cv2 as cv
import numpy as np
import os
import time

# Text that appears at the bottom of the video
caption: list[str] = ["CAPTION TEST"]

# Text that appears above the hand
title: str = "HAND TEST"


def write_cap(img: np.ndarray, img_W: int, img_H: int) -> np.ndarray:
    """Draw the current caption at the bottom of the image."""
    global caption

    if not caption:
        return img

    complete_cap = concate_caption()
    font_size = 0.8
    font_thick = 1
    font_face = cv.FONT_HERSHEY_SIMPLEX
    Wcolor = (255, 255, 255)  # white
    Gcolor = (55, 55, 55)     # gray

    # Caption size
    (text_W, text_H), baseline = cv.getTextSize(
        complete_cap, font_face, font_size, font_thick
    )

    # Caption position
    x_rel_center = (img_W - 1) // 2 - text_W // 2
    y_rel = int((img_H - 1) * 0.9)

    # Caption background box
    CB_min, CB_max = box_pad(
        (x_rel_center, y_rel - text_H - baseline),
        (x_rel_center + text_W, y_rel + baseline),
        1.05,
    )

    img = cv.rectangle(img, CB_min, CB_max, Gcolor, -1)

    return cv.putText(
        img,
        complete_cap,
        (x_rel_center, y_rel),
        font_face,
        font_size,
        Wcolor,
        font_thick,
        cv.LINE_AA,
        bottomLeftOrigin=False,
    )


def append_cap(s: str) -> bool:
    """Append a token to the caption list."""
    caption.append(s)
    return True


def concate_caption() -> str:
    """Concatenate caption tokens into a single string."""
    global caption
    return "".join(s for s in caption)


def clear_cap() -> bool:
    """Clear the caption text."""
    global caption
    caption.clear()
    return True

def save_cap_to_file():
    os.makedirs("captions", exist_ok=True)
    filename = f"captions/caption_{int(time.time())}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(concate_caption())

def write_title(
    img: np.ndarray,
    TL_rect_min: tuple[int, int],
    TL_rect_max: tuple[int, int],
    asl_char: str = "random",
) -> np.ndarray:
    """Draw the title and hand bounding box."""
    global title

    if not title:
        return img

    color = set_title(asl_char)
    x_cord, y_cord = TL_rect_min

    font_size = 0.75
    font_thick = 2
    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)  # white
    margin = 20

    # Title size
    (text_W, text_H), baseline = cv.getTextSize(
        title, font_face, font_size, font_thick
    )

    # Title background box
    TB_min = (x_cord, y_cord - text_H - baseline * 2)
    TB_max = (x_cord + text_W + margin * 2, y_cord)

    # Draw hand bbox
    cv.rectangle(img, TL_rect_min, TL_rect_max, color, 2)

    # Draw title background
    img = cv.rectangle(img, TB_min, TB_max, color, -1)

    return cv.putText(
        img,
        title,
        (x_cord + margin, y_cord - baseline),
        font_face,
        font_size,
        font_color,
        font_thick,
        cv.LINE_AA,
        bottomLeftOrigin=False,
    )


def set_title(label: str) -> tuple[int, int, int]:
    """Set the global title based on label and return a BGR color."""
    global title

    # print(label)
    label = label.lower()

    match label:
        # -------- DIGITS 0–9 --------
        case "0":
            title = "0"
            return (255, 0, 0)
        case "1":
            title = "1"
            return (255, 128, 0)
        case "2":
            title = "2"
            return (255, 255, 0)
        case "3":
            title = "3"
            return (128, 255, 0)
        case "4":
            title = "4"
            return (0, 255, 0)
        case "5":
            title = "5"
            return (0, 255, 128)
        case "6":
            title = "6"
            return (0, 255, 255)
        case "7":
            title = "7"
            return (0, 128, 255)
        case "8":
            title = "8"
            return (0, 0, 255)
        case "9":
            title = "9"
            return (128, 0, 255)

        # -------- LETTERS a–z --------
        case "a":
            title = "A"
            return (5, 91, 219)
        case "b":
            title = "B"
            return (0, 128, 255)
        case "c":
            title = "C"
            return (0, 191, 255)
        case "d":
            title = "D"
            return (0, 255, 255)
        case "e":
            title = "E"
            return (0, 255, 191)
        case "f":
            title = "F"
            return (0, 255, 128)
        case "g":
            title = "G"
            return (0, 255, 64)
        case "h":
            title = "H"
            return (64, 255, 0)
        case "i":
            title = "I"
            return (128, 255, 0)
        case "j":
            title = "J"
            return (191, 255, 0)
        case "k":
            title = "K"
            return (255, 255, 0)
        case "l":
            title = "L"
            return (255, 191, 0)
        case "m":
            title = "M"
            return (255, 128, 0)
        case "n":
            title = "N"
            return (255, 64, 0)
        case "o":
            title = "O"
            return (255, 0, 0)
        case "p":
            title = "P"
            return (255, 0, 64)
        case "q":
            title = "Q"
            return (255, 0, 128)
        case "r":
            title = "R"
            return (255, 0, 191)
        case "s":
            title = "S"
            return (255, 0, 255)
        case "t":
            title = "T"
            return (191, 0, 255)
        case "u":
            title = "U"
            return (128, 0, 255)
        case "v":
            title = "V"
            return (64, 0, 255)
        case "w":
            title = "W"
            return (0, 0, 255)
        case "x":
            title = "X"
            return (0, 64, 255)
        case "y":
            title = "Y"
            return (0, 128, 255)
        case "z":
            title = "Z"
            return (0, 191, 255)

        # -------- FALLBACK --------
        case _:
            title = "?"
            return (128, 128, 128)


def box_pad(
    pt_min: tuple[int, int],
    pt_max: tuple[int, int],
    scale: float = 1.1,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Add padding around a box defined by pt_min and pt_max."""
    x_min, y_min = pt_min
    x_max, y_max = pt_max

    box_W = x_max - x_min
    box_H = y_max - y_min

    pad_W = ((box_W * scale) - box_W) // 2
    pad_H = ((box_H * scale) - box_H) // 2

    new_x_min = int(x_min - pad_W)
    new_y_min = int(y_min - pad_H)
    new_x_max = int(x_max + pad_W)
    new_y_max = int(y_max + pad_H)

    return (new_x_min, new_y_min), (new_x_max, new_y_max)


def check_OOB(pt: tuple[int, int], H: int, W: int) -> tuple[int, int]:
    """Clamp a point to the image bounds [0..W-1] x [0..H-1]."""
    x_cord, y_cord = pt

    if x_cord > (W - 1):
        x_cord = W - 1
    if x_cord < 0:
        x_cord = 0

    if y_cord > (H - 1):
        y_cord = H - 1
    if y_cord < 0:
        y_cord = 0

    return x_cord, y_cord
