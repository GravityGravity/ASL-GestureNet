<!-- Author Images Row -->
<p align="center">
  <img src="figures/fig1.jpg" width="200" height="200"/>
  <img src="figures/fig2.png" width="200" height="200"/>
  <img src="figures/fig3.jpg" width="200" height="200"/>
  <img src="figures/fig4.png" width="200" height="200"/>
</p>

# ✋ ASL-GestureNet


##  Project Overview 

**ASL-GestureNet** is a real-time American Sign Language recognition system that reads **21-point hand keypoints** from a webcam and classifies them into ASL letters.  It runs live on video, predicts gestures frame-by-frame, and shows the recognized character as an on-screen caption.

###### **Authors**
- [GravityGravity](https://github.com/GravityGravity)
- [Howzley](https://github.com/Howzley)
- [mcalvelo28](https://github.com/mcalvelo28)

---

## Model Development & Results

As a team of three, we created our own ASL keypoint dataset by capturing thousands of labeled hand poses using our custom recording tool. We experimented with two model architectures: a 1D **Convolutional Neural Network (CNN)** and a simpler **Multilayer Perceptron (MLP)**.

The CNN quickly learned the training set but showed clear signs of **overfitting**, struggling to generalize to new samples. In contrast, the MLP handled our keypoint-based inputs far better. With proper normalization and training, it achieved **~95% test accuracy** without overfitting.

One remaining challenge is the gestures for **“0” and “O”**, which are visually identical in static ASL. Because both signs share the same hand shape, the model (and even humans) cannot reliably distinguish them from keypoints alone. This ambiguity is expected and noted in our results.


## Project Goals

- **Build a custom ASL dataset** using webcam-captured hand keypoints.
- **Train a lightweight classifier** that predicts static ASL letters from 21-point landmarks.
- **Run real-time gesture recognition** with smooth, low-latency performance.
- **Display live captions** and automatically log recognized sequences.
- **Provide modular tools** for data collection, training, and inference.
- 
<p align="center">
  <img src="\figures\banner.jpg" alt="ASL-GestureNet Banner">
</p>
