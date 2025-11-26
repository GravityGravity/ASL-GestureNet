# FILE: Caption.py
#
# DESC: Gets the most confident word from the confidence matrix and sets the word caption underneath the screen

import cv2 as cv
import numpy as np
caption: list[str] = []


def write_cap(img: np.ndarray):
    pass


def clear_cap():
    caption.clear()
    caption = []
    return True
