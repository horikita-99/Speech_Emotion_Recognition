🎵 Speech Emotion Recognition using CNN + LSTM
This project implements a deep learning pipeline for Speech Emotion Recognition (SER) using the CREMA-D dataset. The model combines Convolutional Neural Networks (CNNs) for extracting spatial features from spectrograms and LSTMs for capturing temporal dependencies in speech, enabling robust classification of human emotions from audio signals.

📌 Dataset: CREMA-D

Corpus of Emotional Multimodal Actors (CREMA-D)

7442 audio clips recorded by 91 actors (48 male, 43 female)

6 emotion categories:

😠 Angry (ANG)

😨 Fear (FEA)

😢 Sad (SAD)

😃 Happy (HAP)

😐 Neutral (NEU)

🤢 Disgust (DIS)
Each audio file is labeled in the filename (e.g., 1079_DFA_ANG_XX.wav).

🔄 Pipeline

Data Preprocessing

Audio resampling to 16kHz mono

Noise reduction & silence trimming

Feature extraction:

MFCCs (40 coefficients + deltas)

Mel-Spectrograms (for CNN input)

Padding to fixed length

Model Architecture

CNN layers → extract frequency & time-based features

LSTM layers → capture sequential dependencies

Dense layers + Softmax → classify into 6 emotions

Training Setup

Loss: categorical_crossentropy

Optimizer: Adam

Metrics: accuracy, f1-score

Regularization: Dropout, BatchNorm, EarlyStopping

Evaluation

Confusion Matrix

Per-class Precision, Recall, F1-score
