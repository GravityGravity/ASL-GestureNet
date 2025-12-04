# FILE: asl_mlp_predict.py
#
# DESC:
#     Loads a trained MLP model and provides predict_asl_mlp() to classify
#     21 hand keypoints (x,y) into ASL a–z / 0–9. Returns '?' on low confidence.
#

import torch
import torch.nn as nn
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder


# ------------------------- Model Definition -------------------------
class ASLMLP(nn.Module):
    """Simple MLP classifier for 42-dim keypoint vectors."""

    def __init__(self, input_size=42, num_classes=36):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ---------------------- Keypoint Normalization ----------------------
def normalize_keypoints(kp):
    """Center on wrist + scale to unit range + flatten."""
    kp = np.array(kp, dtype=np.float32)
    kp -= kp[0]                               # shift wrist to origin
    m = np.abs(kp).max()
    if m > 0:
        kp /= m
    return kp.flatten()


# ------------------------- Load Model + Labels -------------------------
MODEL_PATH = "asl_mlp_model.pth"
model = ASLMLP()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

labels = list("abcdefghijklmnopqrstuvwxyz0123456789")
le = LabelEncoder().fit(labels)


# ----------------------------- Prediction -----------------------------
def predict_asl_mlp(keypoints_input, confidence_threshold=0.1):
    """Predict ASL char from (21,2) keypoints; return '?' if low confidence."""

    # Accept either list or stringified list
    if isinstance(keypoints_input, str):
        keypoints = ast.literal_eval(keypoints_input)
    else:
        keypoints = keypoints_input

    kp = np.array(keypoints, dtype=np.float32)
    if kp.shape != (21, 2):
        raise ValueError(f"Expected (21,2) keypoints, got {kp.shape}")

    x = torch.tensor(normalize_keypoints(kp)).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        max_prob, idx = torch.max(probs, dim=1)

    return le.inverse_transform([idx.item()])[0] if max_prob >= confidence_threshold else '?'
