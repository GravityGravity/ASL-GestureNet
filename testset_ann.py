# FILE: testset_ann.py
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

data_dir = Path.cwd() / "asl_test_dataset.csv"
test_csv_name = 'asl_test_dataset.csv'

test_csv: pd.DataFrame


def csv_check():
    # Dataset directory
    global data_dir
    if not data_dir.exists():
        print(f'{cl.Fore.RED} X {cl.Fore.WHITE} Test_csv DNE: ...Creating new csv file')
        create_csv()
        return test_csv
    print(f'{cl.Fore.GREEN} ✓ {cl.Fore.WHITE} Test_csv already exists')
    return read_test()


def create_csv():
    global test_csv
    test_csv = pd.DataFrame(columns=csv_fieldnames)


def read_test():
    global test_csv
    test_csv = pd.read_csv(test_csv_name)
    return test_csv


def write_csv():
    global test_csv
    test_csv.to_csv('asl_test_dataset.csv',
                    columns=csv_fieldnames, index=False)
    pass


def append_testdata(keypoints: list[tuple[float, float]]):
    global test_csv

    # Ensure we have 21 keypoints
    if len(keypoints) != 21:
        raise IndexError(
            f"\ntestset_ann.py: append_testdata() keypoint list length != 21 (len: {len(keypoints)})\n"
        )

    csv_check()
    keypoints = normalize_scale_KP(keypoints)
    keypoints = upright_KP(keypoints)

    label = input('     > What letter or number class are providing? \n')
    label = label.replace(" ", "")
    test_csv = test_csv.values.tolist()
    new_row = ['row ' + str(len(test_csv)), label] + keypoints
    test_csv.append(new_row)
    test_csv = pd.DataFrame(test_csv, columns=csv_fieldnames)

    print(f'Added new test row with class label \'{label}\' to asl_testset')

    write_csv()

    pass
