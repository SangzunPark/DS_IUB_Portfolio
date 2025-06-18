## FIle DeepLearning_Task_2

1. Speech Denoising with STFT and DNN
Objective:
Restore clean speech from noisy audio using a Denoising Autoencoder approach based on STFT magnitude.

Key Features:
Input: STFT magnitude of noisy audio (|X|)

Target: STFT magnitude of clean speech (|S|)

Model: Dense(256, ELU) → Dense(513, ReLU)

Reconstructs waveform using ISTFT

Evaluates result using SNR (Signal-to-Noise Ratio)

Saves and plays back reconstructed .wav files

Highlights:
Phase is preserved from the noisy input

Two test files (test_x_01.wav, test_x_02.wav) are evaluated


2. Speech Enhancement using 1D CNN
Objective:
Use 1D Convolutional Neural Networks to denoise speech spectrograms.

Key Features:
Input: STFT magnitude of noisy speech

Model:

Conv1D(128) → MaxPool → Conv1D(256) → MaxPool → Flatten → Dense(513)

Activations: ELU + ReLU

He Normal initializer for better convergence

SNR evaluation of reconstructed output

 Highlights:
1D CNN operates along the frequency axis (513 features)

Audio is reconstructed and evaluated in the time domain


3. CIFAR-10 Classification with Data Augmentation
 Objective:
Study the impact of data augmentation on image classification performance.

Key Features:
Base model: 2 Conv layers + Dense layers (simple CNN)

Augmentation methods:

Brighten and darken images

Horizontal flipping

Augmented dataset = 4× original training set

Validation accuracy of original vs. augmented models is visualized

Highlights:
Clear demonstration of augmentation impact using a simple architecture


4. Self-Supervised Pretext Task + Transfer Learning (CIFAR-10)
Objective:
Demonstrate self-supervised learning followed by transfer learning to improve classification on limited data.

Key Features:
Pretext Task:

3 artificial classes: Original, vertically flipped, 90° rotated

CNN trained to classify transformations (self-supervised)

Weights saved to pretext_param.h5

Baseline Model:

Directly trained on only 500 CIFAR-10 samples

Transfer Model:

Reuses pretrained features from Pretext model

Replaces last layer with 10-class output

Uses MultiOptimizer to assign different learning rates:

Pretrained layers: 1e-5

New classifier head: 1e-3

Result Visualization:
Validation accuracy plotted every 100 epochs

Clear comparison: Baseline vs. Transfer Learning

Highlights:
Shows how self-supervised learning can improve performance on small labeled datasets

Transfer model learns more efficiently from limited data

Notes
All experiments are implemented using TensorFlow / Keras

Visualization with matplotlib, audio playback with IPython.display.Audio

Speech tasks use librosa and soundfile, image tasks use CIFAR-10
