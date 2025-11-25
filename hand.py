# FILE: hand.py
#
# DESC: This file tracks hand positions using google's provided media pip library


import os
import sys
import cv2 as cv
import mediapipe as mp


# Capture video stream (Webcam)
cap = cv.VideoCapture(0)


mp_drawing = mp.solutions.drawing_utils

# Load media pipe hand tracking solution
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

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
                        x_min = int(min(x_min, lm.x * frame.shape[1]))
                        y_min = int(min(y_min, lm.y * frame.shape[0]))
                        x_max = int(max(x_max, lm.x * frame.shape[1]))
                        y_max = int(max(y_max, lm.y * frame.shape[0]))
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv.rectangle(frame, (x_min, y_min),
                                 (x_max, y_max), (0, 0, 255), 3)
            cv.imshow('Live Video Feed', frame)  # Display the frame

            # Press 'q' to exit
            if cv.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break
