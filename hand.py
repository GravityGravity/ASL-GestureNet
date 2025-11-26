# FILE: hand.py
#
# DESC: This file tracks hand positions using google's provided media pip library


import os
import sys
import cv2 as cv
import mediapipe as mp
from caption import caption, write_cap, clear_cap

# Capture video stream (Webcam)
cap = cv.VideoCapture(0)

# Load hand keypoint drawing util
mp_drawing = mp.solutions.drawing_utils

# Load media pipe hand tracking solution
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)


def check_OOB(x_cord: int, y_cord: int, H: int, W: int):

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


def box_pad(x_min: int, y_min: int, x_max: int, y_max: int, scale: float = 1.1):
    box_W = (x_max - x_min)
    box_H = (y_max - y_min)
    pad_W = ((box_W * scale) - box_W) // 2
    pad_H = ((box_H * scale) - box_H) // 2

    new_x_min = int(x_min - pad_W)
    new_y_min = int(y_min - pad_H)
    new_x_max = int(x_max + pad_W)
    new_y_max = int(y_max + pad_H)

    return (new_x_min, new_y_min, new_x_max, new_y_max)


if not cap.isOpened():
    print("Error: Could not open video stream or file")
else:
    # Loop to read and display frames
    while (cap.isOpened()):
        success, frame = cap.read()  # Read a frame

        if success:
            RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # H, W, C
            result = hand.process(RGB_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    x_min = float('inf')
                    y_min = float('inf')
                    x_max = float('-inf')
                    y_max = float('-inf')
                    for lm in hand_landmarks.landmark:
                        # Hand Rectangle Boundary Box Coordinates
                        x_min = int(min(x_min, lm.x * (frame.shape[1]) - 1))
                        y_min = int(min(y_min, lm.y * (frame.shape[0]) - 1))
                        x_max = int(max(x_max, lm.x * (frame.shape[1]) - 1))
                        y_max = int(max(y_max, lm.y * (frame.shape[0]) - 1))

                    # Increase padding around hand detection box by about 20%
                    x_min, y_min, x_max, y_max = box_pad(
                        x_min, y_min, x_max, y_max, 1.2)

                    # Check bounds of box to ensure hand detection isnt clipping outside of image
                    x_min, y_min = check_OOB(
                        x_min, y_min, frame.shape[0], frame.shape[1])

                    x_max, y_max = check_OOB(
                        x_max, y_max, frame.shape[0], frame.shape[1])

                    # Draw hand keypoints
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv.rectangle(frame, (x_min, y_min),
                                 (x_max, y_max), (0, 255, 0), 2)
            cv.imshow('Live Video Feed', frame)  # Display the frame

            key = cv.waitKey(100) & 0xFF

            if key == ord('r'):  # record a frame hotkey
                cv.imwrite(frame, ())  # DEBUG/ BUG

                # Press 'q' to exit
            if key == ord('q'):
                break
        else:
            break
