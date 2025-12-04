# FILE: set_create.py
#
#   DESC:
#       Utility module for building a hand-keypoint dataset (train or test) as a CSV.
#
#         - Creating a new CSV with the expected column layout for 21 hand keypoints
#         - Loading an existing CSV and appending new samples
#         - Tracking which ASL character / label the current samples belong to
#         - Normalizing and upright-ing raw MediaPipe keypoints before saving
#
from pathlib import Path
from process import normalize_scale_KP, upright_KP
import colorama as cl
import pandas as pd
import mediapipe as mp
import cv2 as cv

# Column names for the CSV file. Each row corresponds to one hand keypoint
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

# Valid character labels
valid_chars = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
}

# Name of the CSV file we are working with (e.g., "train.csv").
csv_name: str = None

# Path to the CSV file on disk (initialized in csv_check).
data_dir = Path.cwd()

# In-memory representation of the dataset:
#   - While collecting: `test_csv` is a list of rows (list[list[Any]]).
#   - Before saving: it is converted to a DataFrame in `close_csv()`.
test_csv: pd.DataFrame

# The current label / character selected by the user.
selected_char: str


def csv_startup(csv_provided_name: str):
    """
    Entry-point for using this module.

    - Sets the global CSV name
    - Creates or loads the CSV file
    - Prompts the user to choose the character label they are recording
    """
    global csv_name
    csv_name = csv_provided_name
    csv_check(csv_provided_name)
    change_char()
    # Nothing to return; global state is now initialized.


def csv_check(csv_provided_name: str):
    """
    Check if the CSV file already exists.
    - If it does NOT exist: create an empty in-memory CSV structure.
    - If it DOES exist: load it from disk into memory.
    """
    global data_dir
    data_dir = Path.cwd() / csv_provided_name
    print(data_dir)

    if not data_dir.exists():
        # File does not exist yet: start a new dataset.
        print(
            f'{cl.Fore.RED} X {cl.Fore.WHITE} {csv_provided_name} DNE: ...Creating new csv file')
        create_csv()
        return True

    # File already exists: load existing data.
    print(f'{cl.Fore.GREEN} ✓ {cl.Fore.WHITE} {csv_name} already exists, adding to file')
    read_test()
    return True


def char_check(char_to_check: str):
    """
    Validate that the provided character is a single, allowed label.

    Returns:
        True if the character is valid, False otherwise.
    """
    if not len(char_to_check) == 1:
        print(
            f'\n    !!testset_ann.py: \'{char_to_check}\' has length greater or less than 1'
        )
        return False

    if char_to_check not in valid_chars:
        print(
            f'\n    !!testset_ann.py:  \'{char_to_check}\' is not part of valid character list'
        )
        return False

    return True


def change_char():
    """
    Prompt the user to select which character/label they are recording data for.

    This updates the global `selected_char`, and keeps prompting until the user
    enters a valid character.
    """
    global selected_char

    next_char = input(
        '    > Type in char you are providing data for\n        >'
    ).lower()

    while True:
        if char_check(next_char):
            selected_char = next_char
            break
        else:
            next_char = input(
                '    > TRY AGAIN!! Type in char you are providing data for\n        >'
            ).lower()


def create_csv():
    """
    Initialize a brand-new, empty dataset in memory with the proper columns.
    """
    global test_csv
    # Start as an empty DataFrame with the correct column names...
    test_csv = pd.DataFrame(columns=csv_fieldnames)
    # ...but immediately convert to a list of rows for easy appending.
    test_csv = test_csv.values.tolist()


def read_test():
    """
    Load an existing CSV from disk into memory (as a list of rows).
    """
    global test_csv
    global csv_name

    df = pd.read_csv(csv_name)
    test_csv = df.values.tolist()


def clear_csv():
    """
    Reset all global state for this module.

    After calling this, `csv_startup()` must be called again before appending.
    """
    global test_csv
    global csv_name
    global data_dir
    global selected_char

    test_csv = None
    data_dir = None
    csv_name = None
    selected_char = None


def close_csv():
    """
    Convert the in-memory list of rows back to a DataFrame and write it to disk.

    This should be called when you are done collecting samples for the current
    session. It also clears the modules global state afterwards.
    """
    global test_csv
    global csv_name

    # Convert list of rows back to DataFrame with correct column names
    df = pd.DataFrame(test_csv, columns=csv_fieldnames)

    # Write to CSV; `index=False` so we only keep our own "row index" column.
    df.to_csv(csv_name, columns=csv_fieldnames, index=False)

    # Clear globals so a fresh `csv_startup()` can be done later if needed.
    clear_csv()


def append_testdata(keypoints: list[tuple[float, float]]):
    """
    Add one new sample (row) to the in-memory dataset.
    """
    global test_csv
    global selected_char
    global csv_name

    # Ensure we have 21 keypoints
    if len(keypoints) != 21:
        raise IndexError(
            f"\ntestset_ann.py: append_testdata() keypoint list length != 21 (len: {len(keypoints)})\n"
        )

    # Normalize / scale relative to a reference point (e.g., wrist and middle MCP)
    keypoints = normalize_scale_KP(keypoints)

    # Re-orient keypoints so that hand is upright and consistent across samples
    keypoints = upright_KP(keypoints)

    # Build a new row: `row index`, label, then the 21 keypoints
    # - The "row index" is a human-readable string like "row 0", "row 1", ...
    new_row_index = len(test_csv)
    new_row = [f'row {new_row_index}', selected_char] + keypoints

    # Append new sample to our in-memory dataset
    test_csv.append(new_row)

    print(
        f'Added: row {new_row_index} class \'{selected_char}\' to {csv_name}'
    )
