# Use the predict_asl() function to try predicting with the model

import torch
import torch.nn as nn
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

class ASLMLP(nn.Module):
    def __init__(self, input_size=42, num_classes=36):
        super(ASLMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints, dtype=np.float32)
    wrist = keypoints[0]
    keypoints -= wrist
    max_val = np.abs(keypoints).max()
    if max_val > 0:
        keypoints /= max_val
    return keypoints.flatten()

MODEL_PATH = "asl_mlp_model.pth"
model = ASLMLP()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

labels = list("abcdefghijklmnopqrstuvwxyz0123456789")
le = LabelEncoder()
le.fit(labels)

def predict_asl(keypoints_input, confidence_threshold=0.7):
    if isinstance(keypoints_input, str):
        keypoints = ast.literal_eval(keypoints_input)
    else:
        keypoints = keypoints_input

    keypoints = np.array(keypoints, dtype=np.float32)
    if keypoints.shape != (21, 2):
        raise ValueError(f"Expected keypoints of shape (21,2), got {keypoints.shape}")

    input_vector = normalize_keypoints(keypoints)
    input_tensor = torch.tensor(input_vector).unsqueeze(0)  # batch size 1

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)

    if max_prob.item() >= confidence_threshold:
        return le.inverse_transform([pred_idx.item()])[0]
    else:
        return '?'