<p align="center">
  <img src="\figures\banner.jpg" alt="ASL-GestureNet Banner">
</p>

# ✋ ASL-GestureNet

**ASL-GestureNet** is a real-time American Sign Language (ASL) recognition system that uses **hand keypoints** to classify static hand gestures from webcam video. The project extracts 2D hand landmarks, feeds them into a trained neural network classifier, and displays the recognized letter or gesture as an on-screen caption. When no hands are detected, the caption is cleared and automatically logged to a text file.

---

## Project Goals

### **1. Create a Custom ASL Hand Pose Dataset**
- Capture live webcam frames of ASL hand positions.
- Extract 2D hand keypoints (e.g., 21-point landmarks).
- Build a labeled dataset of ASL letters or words for training.

### **2. Train a Gesture Classification Model**
- Use keypoint vectors as input to a neural network classifier.
- Support real-time inference.
- Evaluate accuracy with confusion matrices and per-class metrics.

### **3. Build a Real-Time ASL Recognition Pipeline**
- Detect a hand in webcam frames.
- Extract keypoints per frame.
- Predict gestures instantaneously.
- Display the predicted gesture as a caption on the video feed.

### **4. Implement Caption Logging**
- If no hand is detected for a short time, clear the on-screen caption.
- Save the previously recognized sequence to a `.txt` log file.
- Create a simple transcript of gestures detected during the session.

### **5. Deliver a Demonstrable, High-Usability System**
- Smooth real-time performance.
- Clean, readable UI with live captions.
- Simple controls for starting/stopping the demo.
- Modular code for dataset creation, training, and inference.

---

## Features (High-Level)
- Real-time webcam hand tracking  
- Keypoint-based ASL gesture classification  
- Dynamic on-screen captions  
- Automatic transcript logging  
- Custom dataset recording tool  
- Modular training scripts  

