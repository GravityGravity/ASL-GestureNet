# FILE: hand.py
#
# DESC:
#   Live hand-tracking using MediaPipe Hands. Captures webcam frames,
#   detects a single hand, draws landmarks and bounding boxes, and
#   updates on-screen caption/title.
#

import cv2 as cv
import mediapipe as mp

from caption import write_cap, write_title, box_pad, check_OOB, clear_cap
from process import frame_process


def main() -> None:
    """Run live webcam hand tracking loop."""
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or file")
        return

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(max_num_hands=1)  # single hand tracking

    asl_char = "?"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        H, W = frame.shape[0], frame.shape[1]

        RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)

        hand_coords: list[tuple[float, float]] = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_min = float("inf")
                y_min = float("inf")
                x_max = float("-inf")
                y_max = float("-inf")

                for lm in hand_landmarks.landmark:
                    # normalized (x, y) from MediaPipe
                    cords = (lm.x, lm.y)
                    hand_coords.append(cords)

                    # convert to pixel coords (avoid -1 index)
                    x = lm.x * (W - 1)
                    y = lm.y * (H - 1)

                    x_min = int(min(x_min, x))
                    y_min = int(min(y_min, y))
                    x_max = int(max(x_max, x))
                    y_max = int(max(y_max, y))

            hand_bbox_min = (x_min, y_min)
            hand_bbox_max = (x_max, y_max)

            # add padding around detected hand box
            hand_bbox_min, hand_bbox_max = box_pad(
                hand_bbox_min, hand_bbox_max, 1.2)

            # clamp bbox to image bounds
            hand_bbox_min = check_OOB(hand_bbox_min, H, W)
            hand_bbox_max = check_OOB(hand_bbox_max, H, W)

            # determine if detected hand is left or right
            if (
                result.multi_handedness
                and result.multi_handedness[0].classification[0].label == "Right"
            ):
                is_left = False
            else:
                is_left = True

            # process keypoints in the cropped region (debug / preprocessing)
            frame_process(
                frame,
                hand_bbox_min,
                hand_bbox_max,
                hand_coords,
                is_left=is_left,
                display_hand_seg=True,
            )

            # draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

            # draw title and hand bbox label
            frame = write_title(frame, hand_bbox_min, hand_bbox_max, asl_char)

        # draw bottom caption on the frame
        frame = write_cap(frame, W, H)
        cv.imshow("Live Video Feed", frame)

        key = cv.waitKey(10) & 0xFF

        # clear caption
        if key == ord("x"):
            clear_cap()

        # test changing ASL char label
        if key == ord("a"):
            asl_char = "a"
        if key == ord("b"):
            asl_char = "b"
        if key == ord("c"):
            asl_char = "c"
        if key == ord("d"):
            asl_char = "d"
        if key == ord("e"):
            asl_char = "e"

        # record a frame
        if key == ord("r"):
            cv.imwrite("frame_capture.jpg", frame)  # simple frame capture

        # quit
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
