# FILE: hand.py
#
# DESC: This file tracks hand positions using google's provided media pip library


import os
import sys
import cv2 as cv
import mediapipe as mp
from caption import *

# Capture video stream (Webcam)
cap = cv.VideoCapture(0)

# Load hand keypoint drawing util
mp_drawing = mp.solutions.drawing_utils

# Load media pipe hand tracking solution
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)  # Single Hand Tracking


def frame_process(is_left: bool):
    if not is_left:
        pass
    pass


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

                    hand_bbox_min = (x_min, y_min)
                    hand_bbox_max = (x_max, y_max)
                    hand_bbox_color = (0, 255, 0)

                    # Increase padding around hand detection box by about 20%
                    hand_bbox_min, hand_bbox_max = box_pad(
                        hand_bbox_min, hand_bbox_max, 1.2)

                    # Check bounds of box to ensure hand detection isnt clipping outside of image
                    hand_bbox_min = check_OOB(
                        hand_bbox_min, frame.shape[0], frame.shape[1])

                    hand_bbox_max = check_OOB(
                        hand_bbox_max, frame.shape[0], frame.shape[1])

                    # Draw hand keypoints
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Write Hand Box Title
                    frame = write_title(frame, hand_bbox_min, hand_bbox_color)

                    # Write Hand Bounding Box
                    hand_bbox_color = (0, 255, 0)
                    cv.rectangle(frame, hand_bbox_min,
                                 hand_bbox_max, (0, 255, 0), 2)

                # Process frame
                if result.multi_handedness[0].classification[0].label == 'Left':
                    frame_process(is_left=True)
                if result.multi_handedness[0].classification[0].label == 'Left':
                    frame_process(is_left=False)

            # Write Caption on frame
            frame = write_cap(frame, frame.shape[1], frame.shape[0])
            cv.imshow('Live Video Feed', frame)  # Display the frame

            key = cv.waitKey(1) & 0xFF

            # Clear bottom caption on 'X' button press
            if key == ord('x'):
                clear_cap()

            # Record a frame on 'r' button press
            if key == ord('r'):  # record a frame hotkey
                cv.imwrite(frame, ())  # DEBUG/ BUG

            # Press 'q' to exit
            if key == ord('q'):
                break
        else:
            break
