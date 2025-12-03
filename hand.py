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

from process import frame_process
from caption import write_cap, write_title, box_pad, check_OOB, clear_cap, append_cap, save_cap_to_file
from caption import write_cap, write_title, box_pad, check_OOB, clear_cap, append_cap
import statistics
from collections import deque
import mediapipe as mp
import cv2 as cv
from set_create import append_testdata, change_char, csv_startup, close_csv
from mlp_predictor import predict_asl_mlp
from cnn_predictor import predict_asl_cnn
import os
import warnings
import absl.logging
warnings.filterwarnings(
    "ignore", message="SymbolDatabase.GetPrototype()", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
absl.logging.set_verbosity(absl.logging.ERROR)

record_switch: bool = False


def main() -> None:
    """Run live webcam hand tracking loop."""
    global record_switch

    pred_buffer = deque(maxlen=5)

    kp_loop_count = 0
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("     !!!Error: Could not open video stream or file")
        return

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(max_num_hands=1)  # single hand tracking

    asl_char = "?"

    caption_mode = False
    caption_auto = False
    caption_text = ""
    add_letter_time = 30
    countdown_timer = 0
    last_letter = None
    last_added = None
    clear_caption_time = 100
    space_counter = 0

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

        # if a hand is currently being detected
        if hand_coords:
            # handle model type switching
            if (model_type == 1):
                predicted_label = predict_asl_mlp(hand_coords)
            else:
                predicted_label = predict_asl_cnn(hand_coords)

            pred_buffer.append(predicted_label)

            # smooth out the inputs and write to title
            try:
                smoothed_label = statistics.mode(pred_buffer)
            except statistics.StatisticsError:
                smoothed_label = predicted_label

            frame = write_title(frame, hand_bbox_min,
                                hand_bbox_max, smoothed_label)

        if caption_mode:
            if key == ord('a'):
                caption_auto = not caption_auto
                print(
                    f"CURRENT CAPTION MODE: {'AUTO' if caption_auto else 'MANUAL'}")
                if caption_auto:
                    print(
                        "CAPTION WILL BE UPDATED EVERY SECOND\nAFTER 3 SECONDS OF NO HAND DETECTION THE CAPTION WILL SAVE AND CLEAR")
                else:
                    print(
                        "PRESS [SPACE] TO APPEND THE CURRENT LETTER (SPACE IF NONE) AND [ENTER] TO SAVE AND CLEAR THE CURRENT CAPTION")

            if caption_auto == True:

                if hand_coords:
                    current_letter = smoothed_label
                else:
                    current_letter = " "

                # print(countdown_timer, "   ", current_letter, "   ", last_letter)

                if current_letter == " " and last_letter is None:
                    last_letter = None
                    countdown_timer = 0
                elif current_letter != last_letter:
                    last_letter = current_letter
                    countdown_timer = 0

                if current_letter == last_letter and countdown_timer >= add_letter_time:
                    if current_letter != last_added:
                        if not (current_letter == last_added == ' '):
                            append_cap(current_letter)
                            last_added = current_letter
                        write_cap(frame, W, H)
                        last_letter = current_letter

                if last_letter == current_letter == ' ' and countdown_timer >= clear_caption_time:
                    print("SAVING CURRENT CAPTION TO /captions")
                    save_cap_to_file()
                    clear_cap()
                    last_letter = None
                    countdown_timer = 0

                countdown_timer += 1

            else:
                if hand_coords:
                    current_letter = smoothed_label
                else:
                    current_letter = " "

                if key == ord(' '):
                    append_cap(current_letter)

                write_cap(frame, W, H)

                if key == 13:
                    print("SAVING CURRENT CAPTION TO /captions")
                    save_cap_to_file()
                    clear_cap()
                    last_letter = None

        # draw bottom caption on the frame
        frame = write_cap(frame, W, H)
        cv.imshow("Live Demo Feed", frame)

        key = cv.waitKey(25) & 0xFF

        # # =========CAPTION CONTROLS============
        # # Append to Caption
        # if key == ord('\r'):
        #     append_cap(predicted_label)

        # # Append Space in Caption
        # if not record_switch:
        #     if key == ord(' '):
        #         append_cap(' ')

        # clear caption
        if key == ord("x"):
            clear_cap()

        # toggle caption mode
        if key == ord('v') or key == ord('V'):
            clear_cap()
            caption_mode = not caption_mode
            print(f"CAPTION MODE {'ENABLED' if caption_mode else 'DISABLED'}")
            if caption_mode:
                print("CURRENT TYPE: MANUAL\nTO TOGGLE AUTO-CAPTION PRESS [A]")
            if caption_auto:
                print(
                    "CAPTION WILL BE UPDATED EVERY SECOND\nAFTER 3 SECONDS OF NO HAND DETECTION THE CAPTION WILL SAVE AND CLEAR")
            else:
                print(
                    "PRESS [SPACE] TO APPEND THE CURRENT LETTER (SPACE IF NONE) AND [ENTER] TO SAVE AND CLEAR THE CURRENT CAPTION")
            countdown_timer = 0
            last_letter = None

        # Enable frame recording hotkeys
        if key == ord("r") or key == ord("R"):
            if not record_switch:
                record_switch = True
                kp_loop_count = 0
                print('     \'R\' pressed -> FRAME RECORDING BUTTON ENABLED')
                data_csv_name: str = input(
                    '    > Provide name of csv you want to add to or create\n        >')
                csv_startup(data_csv_name)
            else:
                print('     \'R\' -> FRAME RECORDING BUTTON ALREADY ENABLED')

        # Disable frame recording hotkeys
        if (record_switch and (key == ord("c") or key == ord("C"))):
            record_switch = False
            close_csv()
            print('     \'C\' -> FRAME RECORDING BUTTON DISABLED\n'
                  '                     SAVED CSV TO FILE :) ')

        # Hotkey to record a single frame's keypoint data
        if record_switch:
            if key == ord(' '):
                if hand_coords:
                    append_testdata(hand_coords)
                    kp_loop_count += 1

            if key == ord('f') or key == ord('F'):
                kp_loop_count = 0
                change_char()

            if kp_loop_count > 25:
                print('    \n==25 DATA POINTS CREATED FOR CLASS, INPUT NEXT_CHAR==\n')
                kp_loop_count = 0
                change_char()

        # ============QUIT PROGRAM==============
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
