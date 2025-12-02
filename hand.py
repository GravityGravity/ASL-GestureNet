# FILE: hand.py
#
# DESC:
#   Live hand-tracking using MediaPipe Hands. Captures webcam frames,
#   detects a single hand, draws landmarks and bounding boxes, and
#   updates on-screen caption/title.
#


# HOTKEYS:
#   [R]  -> Enable frame recording mode (prompts for CSV file name)
#   [C]  -> Disable frame recording mode and save CSV to disk
#   [SPACE] -> Capture and record a single frame's keypoint data (when recording is enabled)
#   [F]  -> Change the active character label for recorded samples (when recording is enabled)
#   [X]  -> Clear the on-screen caption
#   [Q]  -> Quit the program (auto-saves CSV if recording is active)

from cnn_predictor import predict_asl_cnn
from mlp_predictor import predict_asl_mlp
from testset_ann import append_testdata
from key_points_predictor import predict_asl
from set_create import append_testdata, change_char, csv_startup, close_csv
import cv2 as cv
import mediapipe as mp

from caption import write_cap, write_title, box_pad, check_OOB, clear_cap
from process import frame_process
record_switch: bool = False


def main() -> None:
    """Run live webcam hand tracking loop."""
    global record_switch
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or file")
        return

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(max_num_hands=1)  # single hand tracking

    asl_char = "?"

    # await model type selection
    model_type = None
    while model_type not in ("1", "2"):
        model_type = input(
            "What model would you like to use (MLP - 1, CNN - 2): ")
    model_type = int(model_type)

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
                    cords = [lm.x, lm.y]
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

        # handle model type switching
        if hand_coords:
            if (model_type == 1):
                # write the predicted label to the title for MLP
                predicted_label = predict_asl_mlp(hand_coords)
                frame = write_title(frame, hand_bbox_min,
                                    hand_bbox_max, predicted_label)
            elif (model_type == 2):
                predicted_label = predict_asl_cnn(hand_coords)
                frame = write_title(frame, hand_bbox_min,
                                    hand_bbox_max, predicted_label)
            else:
                return

        # draw bottom caption on the frame
        frame = write_cap(frame, W, H)
        cv.imshow("Live Demo Feed", frame)

        key = cv.waitKey(25) & 0xFF

        # clear caption
        if key == ord("x"):
            clear_cap()

        # Enable frame recording hotkeys
        if key == ord("r") or key == ord("R"):
            if not record_switch:
                record_switch = True
                print('     \'R\' pressed -> FRAME RECORDING BUTTON ENABLED')
                data_csv_name: str = input(
                    '    > Provide name of csv you want to add to or create\n        >')
                csv_startup(data_csv_name)
            else:
                print('     \'R\' -> FRAME RECORDING BUTTON ALREADY ENABLED')

        # Disable frame recording hotkeys
        if key == ord("c") or key == ord("C"):
            record_switch = False
            close_csv()
            print('     \'C\' -> FRAME RECORDING BUTTON DISABLED\n'
                  '                     SAVED CSV TO FILE :) ')

        # Hotkey to record a single frame's keypoint data
        if record_switch:
            if key == ord(' '):
                if hand_coords:
                    append_testdata(hand_coords)

            if key == ord('f') or key == ord('F'):
                change_char()

        # quit
        if key == ord("q") or key == ord("q"):
            if record_switch:
                close_csv()
                print('     ...\'Q\' -> SAVED CSV FILE\n'
                      '                     EXITING PROGRAM ... ')
                break
            print('     ...\'Q\' -> EXITING PROGRAM ... ')
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
