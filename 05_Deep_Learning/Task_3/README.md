1. Model Compression using SVD (MNIST)
Goal:
Compress a fully connected MNIST classifier using Singular Value Decomposition (SVD) on trained weights.

Approach:
Trained a deep MLP: Flatten → 5×Dense(1024, ReLU) → Dense(10, Softmax)

Performed SVD on hidden layers’ weight matrices

Reconstructed lower-rank approximations with varying D values (e.g., D=10~200)

Compared model accuracy vs. parameter count

Outcome:
High compression (e.g., D=50) retained most of the accuracy while reducing model size significantly.


2. Factorized Dense Layers with SVD (MNIST)
Goal:
Instead of modifying weights post-training, rebuild the model using SVD-decomposed Dense layers.

Approach:
Each Dense(1024) is replaced with:

scss
Dense(D) → Dense(1024)
U, Σ, Vᵗ from SVD used to initialize both layers

Fixed-rank (D=20) used across all layers

Outcome:
Efficient compression built into the architecture itself; retraining for 10 epochs achieved strong results.


3. SVD-Constrained Training (MNIST)
Goal:
Train a deep network while maintaining low-rank SVD structure during training.

Approach:
After each epoch, compress all layer weights using SVD (D=20)

Forward & backward pass uses compressed weights

Weight updates reflect constraints throughout training

Outcome:
Shows that accurate training is possible even under continuous compression, not just post hoc.


4. Speaker Verification with Siamese GRU Network
Goal:
Determine whether two utterances are from the same speaker using Siamese architecture and GRU encoders.

Approach:
Compute STFT magnitudes for all utterances

Generate positive/negative speaker pairs

Siamese model:

less
Input A/B → Shared GRU → Dense layers → |A−B| → Sigmoid
Trained with BinaryCrossentropy on paired data

Outcome:
Achieved strong speaker discrimination accuracy by learning speaker embeddings via GRU and pairwise distance.


5. Speech Denoising via IBM Mask Estimation with GRU
Goal:
Denoise noisy speech spectrograms by learning to predict an Ideal Binary Mask (IBM).

Approach:
Input: magnitude STFT of noisy speech

Target: IBM (1 if speech > noise, else 0)

Model:

scss
Masking → GRU(512) → Dropout → GRU(256) → TimeDistributed(Dense(513, sigmoid))
Predicted masks applied to noisy input → ISTFT → denoised waveform

Outcome:
Achieved notable SNR improvements on validation data. Denoised .wav files recovered and saved.