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
            RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = hand.process(RGB_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    print(hand_landmarks)
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands)

            cv.imshow('Live Video Feed', frame)  # Display the frame

            # Press 'q' to exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
