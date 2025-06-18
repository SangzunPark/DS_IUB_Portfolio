# Deep Learning MNIST & Audio Projects

This repository contains a collection of deep learning experiments across image and audio domains, with a focus on the MNIST dataset and speech enhancement tasks. Each module explores different architectures and objectives ranging from supervised classification to generative modeling.

---

## Included Projects

### 1. **Softmax MNIST Classifier**
- **Type**: Baseline MLP
- **Task**: Classifies handwritten digits using softmax and cross-entropy.
- **Architecture**: Fully connected layers, ReLU activations.
- **Evaluation**: Accuracy on the MNIST test set.

---

### 2. **Autoencoder for Speech Denoising**
- **Type**: 1D Convolutional Autoencoder
- **Task**: Speech enhancement by predicting clean spectrograms from noisy ones.
- **Evaluation**: Reconstructed audio and SNR comparison.

---

### 3. **Shallow Neural Network**
- **Task**: MNIST classification with a shallow fully connected network.
- **Highlights**: Custom backpropagation implementation from scratch.
- **Goal**: Educational purpose to understand layer-wise gradient flows.

---

### 4. **Full Backpropagation Networks**
- **Task**: Manual implementation of full gradient descent and backprop.
- **Focus**: Deeper analysis of layer-wise updates and activation derivatives.

---

### 5. **Low-Rank Approximation with SVD**
- **Type**: SVD compression on dense layers
- **Task**: Apply low-rank factorization to reduce model size.
- **Results**: Comparison of accuracy and parameter count under different ranks.

---

### 6. **Variational AutoEncoder (VAE)**
- **Task**: Generative model on MNIST
- **Features**: Latent traversal, KL divergence regularization
- **Visuals**: Generated samples and latent dimension manipulation.

---

### 7. **Conditional GAN (cGAN)**
- **Task**: Digit generation conditioned on labels.
- **Evaluation**: Class-specific sample generation and discriminator accuracy plots.

---

### 8. **Center-Patch Conditional GAN**
- **Input**: Center 10×10 patch + noise
- **Task**: Generate full image conditioned on local patch.
- **λ-Controlled Regularization**: Adjusts fidelity to the center patch.

---

### 9. **LSTM-based Patch Completion**
- **Input**: Upper 8 patches of an image
- **Output**: Predicts lower 8 patches via LSTM
- **Use case**: Image inpainting and sequential generation.

---

## External Projects (in other repositories)
- https://github.com/SangzunPark/Deep_Learning

##  Author

**Sangzun Park**
Graduate Student, MS in Data Science  
Indiana University Bloomington  