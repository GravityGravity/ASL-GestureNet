# FILE: process.py
#
# DESC:
#   Utilities for processing MediaPipe hand keypoints. This includes:
#     • Converting normalized keypoints into crop-local coordinates
#     • Centering keypoints on the wrist
#     • Scaling based on wrist → middle_finger knuckle distance
#     • Rotating the hand so the middle finger points vertically everytime
#   Used to generate consistent, normalized keypoint data for training, testing, and live demo.
#
import numpy as np
import cv2 as cv


def frame_process(
    img: np.ndarray,
    min_pt: tuple[int, int],
    max_pt: tuple[int, int],
    keypoints: list[tuple[float, float]],
    is_left: bool,
    display_hand_seg: bool = False,
) -> list[list[float]]:
    """Process raw hand keypoints into cropped, normalized, upright coordinates."""

    # Ensure we have 21 keypoints
    if len(keypoints) != 21:
        raise IndexError(
            f"\nprocess.py: frame_process() keypoint list length != 21 (len: {len(keypoints)})\n"
        )
    # Convert input keypoints to numpy array
    kp_array = np.asarray(keypoints, dtype=np.float32)

    # Unpack bounding box points
    x_min, y_min = min_pt
    x_max, y_max = max_pt

    # Get image height and width
    H, W = img.shape[0], img.shape[1]

    # Mirror keypoints and bbox if this is a right hand
    if not is_left:
        kp_array[:, 0] = 1.0 - kp_array[:, 0]  # flip normalized x
        left_x_min = W - x_max - 1            # flipped bbox min x
        left_x_max = W - x_min - 1            # flipped bbox max x
        x_min, x_max = left_x_min, left_x_max

    # Convert normalized coords to original image pixel coords
    kp_org = kp_array * np.asarray([W, H], dtype=np.float32)

    # Compute crop width and height in pixels
    crop_W = (x_max - x_min)
    crop_H = (y_max - y_min)

    # Guard against zero-size crops
    if crop_W <= 0 or crop_H <= 0:
        raise ValueError(
            f"frame_process: invalid crop size ({crop_W}, {crop_H})")

    # Shift keypoints into crop-local pixel coordinates
    kp_new = kp_org - np.asarray([x_min, y_min], dtype=np.float32)

    # Normalize keypoints to [0, 1] within the crop
    kp_new = kp_new / np.asanyarray([crop_W, crop_H], dtype=np.float32)

    # Convert to list for downstream functions
    kp_new_list = kp_new.tolist()

    # Center and scale keypoints relative to wrist and middle MCP
    kp_new_list = normalize_scale_KP(kp_new_list)

    # Rotate keypoints so middle MCP is vertical
    kp_new_list = upright_KP(kp_new_list)

    # print(kp_new_list)  # debug

    # Optionally display the hand crop
    if display_hand_seg:
        if not is_left:
            img = hand_invert(img)  # horizontally flip the image

        # print((y_min, y_max, x_min, x_max)) # Debug
        img = img[y_min:y_max + 1, x_min:x_max + 1]  # crop image
        cv.imshow("hand_segment", img)

    # Return processed keypoints as list of [x, y]
    return kp_new_list


def hand_invert(img: np.ndarray) -> np.ndarray:
    """Flip an image horizontally."""
    return cv.flip(img, 1)  # 1 = horizontal flip


def upright_KP(keypoints: list[tuple[float, float]]) -> list[list[float]]:
    """Rotate keypoints so the wrist→middle_MCP vector points vertically."""
    # Type check input container
    if not isinstance(keypoints, list):
        raise TypeError(
            f"\nprocess.py: UPRIGHT_KP() type not taken {type(keypoints)}\n"
        )

    # Ensure we have 21 keypoints
    if len(keypoints) != 21:
        raise IndexError(
            f"\nprocess.py: UPRIGHT_KP() keypoint list length != 21 (len: {len(keypoints)})\n"
        )

    # Convert keypoints to numpy array
    np_keypoints = np.array(keypoints, dtype=np.float32)  # (21, 2)

    # Wrist landmark (unused but may be useful later)
    wrist = np_keypoints[0]  # (2,)

    # Middle MCP landmark (index 9)
    vec = np_keypoints[9]  # (2,)
    vx, vy = vec[0], vec[1]

    # Angle of current wrist→middle_MCP vector
    angle = np.arctan2(vy, vx)

    # Target angle for vertical (negative y direction)
    target = -np.pi / 2

    # Rotation angle needed
    theta = target - angle

    # Build 2D rotation matrix
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    # Apply rotation to all keypoints
    rotated = np_keypoints @ R.T  # (21, 2)

    # Zero-out tiny numerical noise
    rotated[np.abs(rotated) < 1e-6] = 0.0

    # Debug print of rotated keypoints
    # print(rotated.tolist())

    # Return rotated keypoints as list of [x, y]
    return rotated.tolist()


def normalize_scale_KP(keypoints: list[tuple[float, float]]) -> list[list[float]]:
    """Center keypoints at wrist and scale so wrist→middle_MCP has length 1."""
    # Type check input container
    if not isinstance(keypoints, list):
        raise TypeError(
            f"\nprocess.py: NORMALIZE_SCALE_KP() type not taken {type(keypoints)}\n"
        )

    # Ensure we have 21 keypoints
    if len(keypoints) != 21:
        raise IndexError(
            f"\nprocess.py: NORMALIZE_SCALE_KP() keypoint list length != 21 (len: {len(keypoints)})\n"
        )

    # Convert keypoints to numpy array
    np_keypoints = np.array(keypoints, dtype=np.float32)  # (21, 2)

    # Wrist landmark (index 0)
    wrist = np_keypoints[0]  # (2,)

    # Shift all points so wrist is at origin
    center_wrist_KP = np_keypoints - wrist  # (21, 2)

    # Distance from wrist to middle MCP (index 9)
    dist = np.linalg.norm(center_wrist_KP[9])

    print(dist)

    # Handle degenerate case where distance is too small
    if dist < 1e-4:
        print('   \n BAD DIST!!!!!!!!')
        # Return sentinel large value for all points (shape 21x2)
        return [[999.0, 999.0] for _ in range(21)]

    # Scale all points so wrist→middle_MCP has unit length
    scaled_CW_KP = center_wrist_KP / dist  # (21, 2)

    # print(scaled_CW_KP.tolist())  # Debug

    # Return scaled keypoints as list of [x, y]
    return scaled_CW_KP.tolist()
