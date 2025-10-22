# GestureFace-Recognition-System-GFRS-
GFRS is designed to recognize and compare human hand positions and facial expressions in images. By leveraging hand and facial keypoint tracking along with trained machine learning models, GFRS identifies expressive and gestural similarities across different images.


### Our Problem
Our team is focusing on the problem of recognizing and comparing human hand positions and facial expressions in images. Accurately identifying these visual cues is challenging due to variations in lighting, angles, occlusions, and individual differences in appearance and expression. Additionally, combining both facial and hand cues presents complexity in distinguishing subtle emotional and gestural patterns across different images.


### Solution
To address this problem, our team will use machine learning models trained on custom datasets that include various hand positions and facial expressions. By creating our own dataset, we can ensure diverse and representative examples for accurate recognition.

Our solution will incorporate hand keypoint tracking and facial keypoint tracking techniques to detect and analyze the spatial relationships of key features. These keypoints will be used to represent gestures and expressions numerically, allowing the system to compare and match images with similar facial expressions and hand positions.

Through this approach, we aim to build a model capable of identifying both emotional and physical similarities across different images with high precision.


### Model Evaluation
We will evaluate our trained models using separate validation and testing datasets to measure performance and generalization. The validation dataset will be used during training to fine-tune model parameters and prevent overfitting, while the testing dataset will provide an unbiased assessment of the model’s accuracy and reliability. Evaluation metrics such as precision, recall, F1-score, and mean squared error on keypoint detection will be used to assess how effectively the model recognizes and matches hand positions and facial expressions across unseen images.



#### Library Requirements + Purpose

Requirements.txt contains all library requirements to run this project

- Pandas: Extracting CSV file information
- MediaPipe (By google): 3D/2D hand tracking points 
- Torch/TorchVision: Training models and creating datasets/dataset loaders