<p align="center">
  <img src="\figures\banner.jpg" alt="ASL-GestureNet Banner">
</p>

# ✋ ASL-GestureNet


##  Project Overview 

**ASL-GestureNet** is a real-time American Sign Language recognition system that reads **21-point hand keypoints** from a webcam and classifies them into ASL letters.  It runs live on video, predicts gestures frame-by-frame, and shows the recognized character as an on-screen caption.

###### **Authors**
- [GravityGravity](https://github.com/GravityGravity)
- [Howzley](https://github.com/Howzley)
- [mcalvelo28](https://github.com/mcalvelo28)

---

##  Features
- **Real-time webcam hand tracking** using MediaPipe  
- **Keypoint-driven gesture classification** powered by a custom neural network  
- **Live on-screen captions** that update instantly with each prediction  
- **Automatic transcript logging** whenever the hand leaves the frame  
- **Built-in dataset recording tool** for capturing and labeling ASL poses  
- **Modular training + inference pipeline**, easy to extend and experiment with  

---

## 🎯 Project Goals (Detailed)
### **1. Create a Custom ASL Hand Pose Dataset**
- Capture ASL hand images directly from a webcam.  
- Extract and store 2D keypoints for each gesture.  
- Build a labeled dataset suitable for training the model.

### **2. Train a Gesture Classification Model**
- Use normalized hand-landmark vectors as input.  
- Train a fast classifier for static ASL letters.  
- Evaluate with confusion matrices and per-class metrics.

### **3. Build a Real-Time Recognition Pipeline**
- Detect one hand per frame.  
- Extract keypoints, run inference, and show the prediction.  
- Update the caption continuously as gestures change.

### **4. Implement Caption Logging**
- Clear the caption when no hand is detected.  
- Save the previous sequence to a text file.  
- Produce a simple session transcript.

### **5. Deliver a Smooth, Usable System**
- Low-latency performance.  
- Clean UI with readable captions.  
- Easy to run demos and experiments.  
- Organized modules for dataset creation, training, and inference.

---
