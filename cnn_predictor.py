import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

class HandKeypointCNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 21, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 21, 2) -> (B, 2, 21)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
MODEL_PATH = "asl_cnn_model.pth"
model = HandKeypointCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

labels = list("abcdefghijklmnopqrstuvwxyz0123456789")
le = LabelEncoder()
le.fit(labels)

def predict_asl_cnn(keypoints, confidence_threshold = 0):
    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)  # (1, 21, 2)
    input_tensor = input_tensor.to('cpu')

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)

    if max_prob.item() >= confidence_threshold:
        return le.inverse_transform([pred_idx.item()])[0]
    else:
        return '?'