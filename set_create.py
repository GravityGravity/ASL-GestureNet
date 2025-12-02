# FILE: set_create.py
#
#   DESC:
#       This code creates and appends to a test_dataset csv file used for model evaluation
#

from pathlib import Path
from process import normalize_scale_KP, upright_KP
import colorama as cl
import pandas as pd
import mediapipe as mp
import cv2 as cv

csv_fieldnames = [
    "row index",
    "label",
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

valid_chars = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
}
csv_name: str = None
data_dir = Path.cwd()

test_csv: pd.DataFrame
selected_char: str


def csv_startup(csv_provided_name: str):
    global csv_name
    csv_name = csv_provided_name
    csv_check(csv_provided_name)
    change_char()
    pass


def csv_check(csv_provided_name: str):
    # Dataset directory
    global data_dir
    data_dir = Path.cwd() / csv_provided_name
    print(data_dir)
    if not data_dir.exists():
        print(f'{cl.Fore.RED} X {cl.Fore.WHITE} Test_csv DNE: ...Creating new csv file')
        create_csv()
        return True
    print(f'{cl.Fore.GREEN} ✓ {cl.Fore.WHITE} {csv_name} already exists, adding to file')
    read_test()
    return True


def char_check(char_to_check: str):
    if not len(char_to_check) == 1:
        print(
            f'\n    !!testset_ann.py: \'{char_to_check}\'has length greater or less than 1')
        return False

    if not char_to_check in valid_chars:
        print(
            f'\n    !!testset_ann.py:  \'{char_to_check}\' is not part of valid character list')
        return False

    return True


def change_char():
    global selected_char
    next_char = input(
        '    > Type in char you are providing data for\n        >')
    next_char = next_char.lower()
    while (True):
        if char_check(next_char):
            selected_char = next_char
            break
        else:
            next_char = input(
                '    > TRY AGAIN!! Type in char you are providing data for\n        >')
            next_char = next_char.lower()


def create_csv():
    global test_csv
    test_csv = pd.DataFrame(columns=csv_fieldnames)
    test_csv = test_csv.values.tolist()


def read_test():
    global test_csv
    global csv_name
    test_csv = pd.read_csv(csv_name)
    test_csv = test_csv.values.tolist()


def clear_csv():
    global test_csv
    global csv_name
    global data_dir
    global selected_char

    test_csv = None
    data_dir = None
    csv_name = None
    selected_char = None


def close_csv():
    global test_csv
    global csv_name
    test_csv = pd.DataFrame(test_csv, columns=csv_fieldnames)
    test_csv.to_csv(csv_name,
                    columns=csv_fieldnames, index=False)
    clear_csv()
    pass


def append_testdata(keypoints: list[tuple[float, float]]):
    global test_csv
    global selected_char
    global csv_name

    # Ensure we have 21 keypoints
    if len(keypoints) != 21:
        raise IndexError(
            f"\ntestset_ann.py: append_testdata() keypoint list length != 21 (len: {len(keypoints)})\n"
        )

    keypoints = normalize_scale_KP(keypoints)
    keypoints = upright_KP(keypoints)
    new_row = ['row ' + str(len(test_csv)), selected_char] + keypoints
    test_csv.append(new_row)

    print(
        f'Added: row {str(len(test_csv))} class \'{selected_char}\' to {csv_name}')
