Goal
To explore multiple generative and predictive modeling techniques on the MNIST dataset, including:

Sequential patch-based LSTM prediction

Variational AutoEncoder (VAE)

Conditional GAN (cGAN)

Center-conditioned GAN with regularization


1. Patch-based LSTM Image Generation
MNIST images split into 16 non-overlapping 7×7 patches.

LSTM model trained to predict future patches given previous ones (auto-regressive).

Only the upper half (first 8 patches) is input; the model generates the lower half (next 8 patches).

Evaluation:

Trained using MSE

Visual comparison of generated and original images.


2. Variational AutoEncoder (VAE)
Encoder: 2-layer CNN → latent space z ~ N(μ, σ)

Decoder: Dense + Conv2DTranspose to reconstruct 28×28 images

Loss = Binary Cross Entropy + KL Divergence

Visualizations:

Reconstruction of test images

Latent space traversal (variation along each latent dimension)

Latent variation visualization using real test images


3. Conditional GAN (cGAN)
Generator and Discriminator both conditioned on class labels (0-9).

Generator input: z (random noise) + label embedding

Discriminator input: image + label embedding

Trains using Binary Cross-Entropy

Visual output:

Class-conditional digit generation

Accuracy plot of discriminator’s real vs. fake prediction over epochs


4. Center-Patch Conditional GAN
Goal: Generate a full 28×28 image given:

Random noise (100-dim)

A flattened center 10×10 patch (100-dim)

Generator is trained with both adversarial loss and MSE loss between real/fake center patches

Regularization hyperparameter λ tested at 0.1 and 10

Output:

Two sets of images generated for each λ

Higher λ → center-patch constraints are stronger (more faithful)