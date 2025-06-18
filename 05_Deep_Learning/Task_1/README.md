## FIle DeepLearning_Task_1

1. Softmax MNIST Classification Problem
Code 1:

Single-layer softmax classifier

Visualizes weights at epochs 20 and 200

Tracks and plots test accuracy over epochs

2. Autoencoder-Based Feature Learning Problem
Code 2: Naive Autoencoder

Code 3: Sparse Autoencoder (with KL Divergence regularization)

Code 4: Classifier comparison (No feature / Naive / Sparse) — encoder weights frozen

Code 5: Classifier comparison extended with NN — encoder weights trainable

These 4 codes form one cohesive experimental sequence focused on learning features using autoencoders and applying them to downstream classification.

3. Shallow Neural Network (Single Hidden Layer) Problem
Code 6:

Standard shallow network with 1 hidden sigmoid layer

Compares performance with/without pre-trained encoder weights

4. Full Backpropagation on Deep Layers (Init & BN Experiments)
Code 7: Deep network with various weight initializations & activations — no BatchNorm

Code 8: Same structures as above but with Batch Normalization applied

These two codes focus on investigating how initialization strategies and BatchNorm affect training and accuracy in deep networks.



