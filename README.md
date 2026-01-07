# Pixel-Play-26-Harsh-SIngh
Introduction
This project aims to develop a model to detect anomalous events in surveillance footage. The task poses unique challenges, particularly the requirement to accurately identify irregularities using the provided video dataset. The task involves training a model to analyze sequential video frames and assign probability scores to potential anomalies. The focus is on generalization and the ability to distinguish between standard background activity and significant deviations in a real-world environment

Model Development

Model 1 – HOG + Optical Flow + Isolation Forest

Data Preprocessing

Image preprocessing
All frames were converted from RGB to grayscale to reduce computational cost and noise. Frames were resized to 96×96 to balance motion preservation and efficiency. Histogram normalization was applied to HOG features to stabilize gradients.

Data restructuring and handling
Frames were processed sequentially for each video to preserve temporal order. Each video was handled independently to avoid mixing temporal information across videos. Previous frames were explicitly stored to compute optical flow features.

Model Architecture

Appearance features (HOG)
Histogram of Oriented Gradients was used to capture edge and shape information. Due to the very high dimensionality of raw HOG features, PCA was applied to retain the 256 most informative components.

Motion features (Optical Flow)
Dense optical flow was used to capture frame-to-frame motion. Statistical summaries of motion magnitude were extracted to represent motion intensity and variation.

Feature fusion
HOG and optical flow features were concatenated and standardized using a StandardScaler fitted only on training data to prevent data leakage.

Anomaly detection
Isolation Forest was used to model normal behavior. Frames that were isolated quickly in the trees were assigned higher anomaly scores.

Training 
Isolation Forest was trained only on training frames assuming all training data represents normal behavior. No explicit loss function or optimizer was used. PCA and scaling were fitted strictly on training data. The model was trained on a Kaggle environment.

Model 2 – Random ResNet-18 + Isolation Forest

Data Preprocessing

Image preprocessing
Frames were converted to RGB tensors and resized to 32×32. ImageNet-style normalization was applied to ensure stable numerical ranges during convolution operations.

Data restructuring and handling
Frames were processed video-wise. CNN features were extracted independently for each frame while preserving video boundaries.

Model Architecture

Deep feature extractor
A ResNet-18 architecture was used with its final classification layer removed. The network produced 512-dimensional feature vectors per frame. The network weights were randomly initialized and kept frozen.

Feature interpretation
Although the weights were random, the convolutional structure preserved spatial locality and produced consistent projections across frames, making the features useful for anomaly separation.

Anomaly detection
Isolation Forest was trained on the extracted CNN features to model normal visual patterns.

Training

The ResNet model was used only for inference in evaluation mode. Isolation Forest was trained on CNN features extracted from training videos. Temporal smoothing using a rolling mean was applied to reduce noisy frame-level predictions.

Methods Tried and Intuition

Model 1 relies on handcrafted appearance and motion features, making it sensitive to low-level changes. Model 2 relies on deep convolutional structure to project frames into a stable high-dimensional space. Although the CNN was not pretrained, its fixed random filters acted as a consistent feature encoder. The ensemble of both models worked well because each captured different aspects of anomalies: motion irregularities and visual inconsistencies.

Results

Model 1 achieved an AUC score of approximately 0.50. Model 2 achieved an AUC of around 0.60 after temporal smoothing. A weighted ensemble of both models produced the best performance with an AUC of approximately 0.63.

Conclusion

The results show that combining classical computer vision features with deep CNN-based representations improves anomaly detection performance. The ensemble approach reduced false negatives and produced more confident anomaly scores. Even without pretrained weights, the CNN-based model contributed useful structure to the system.

Challenges and Solutions

At the beginning, I tried using a One-Class SVM model, but it did not work well for this problem because the data was complex and highly variable. Because of this, I switched to Isolation Forest, which gave better and more stable results. Another challenge was understanding how random CNN features could still be useful. Since I did not have deep knowledge of neural networks, I avoided using very complex deep learning techniques and instead focused on using CNNs only as fixed feature extractors. The frame-level anomaly scores were often noisy due to sudden motion changes, which was handled by applying temporal smoothing. Working with high-dimensional feature vectors was challenging, but Isolation Forest handled them well and improved overall performance.

Learning Outcomes

* Learned to build an end-to-end anomaly detection pipeline
* Gained experience with feature engineering and preprocessing
* Understood how CNN architectures can be used as feature extractors
* Learned practical debugging and iterative improvement strategies
* Developed confidence in handling models when initial approaches fail
