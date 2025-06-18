#!/usr/bin/env python
# coding: utf-8

# ## Speech Denoising with STFT and DNN

# I first encountered STFT (Short-Time Fourier Transform) through this problem. I managed to solve the problem following the professor's instructions, but I still have some questions about STFT. Particularly, I have doubts about concepts like time resolution, overlap, their interrelationship, and the computation method, among other things.

# In[2]:


import librosa
# STFT(Spectrogram) calculation
s, sr=librosa.load('train_clean_male.wav', sr=None)
S=librosa.stft(s, n_fft=1024, hop_length=512)
sn, sr=librosa.load('train_dirty_male.wav', sr=None)
X=librosa.stft(sn, n_fft=1024, hop_length=512)


# In[4]:


import numpy as np
import tensorflow as tf
import soundfile as sf

# Apply abs
abs_s = np.abs(S)
abs_x = np.abs(X)

# Moedel experiments, the values of last layer should not be a negative ; relu
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(513,)),
    #tf.keras.layers.Dense(256, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='elu'),
    #tf.keras.layers.Dropout(0.5), 
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(513, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dense(513, activation='elu'),
])

model.compile(optimizer = 'adam', loss='mean_squared_error')

model.fit(abs_x.T, abs_s.T, epochs=40, batch_size=128)

# Load test data and apply STFT
test_noisy, sr = librosa.load('test_x_01.wav', sr=None)
test_x = librosa.stft(test_noisy, n_fft=1024, hop_length=512)

# Predict test x
test_abs_x = np.abs(test_x)
p_abs_s = model.predict(test_abs_x.T).T

#Recover the (complex-valued) speech spectrogram of the test signal
p_hat_s = p_abs_s * (test_x / (test_abs_x)) # + 1e-20

# Reconstructin test file
sh_test = librosa.istft(p_hat_s, hop_length=512)
sf.write('test_s_01_recons.wav', sh_test, sr)

# Compare with ground truth
ground_truth, sr = librosa.load('test_s_01.wav', sr=None)

# Adjust length between two files
if sh_test.shape[0] > ground_truth.shape[0]:
    sh_test = sh_test[:ground_truth.shape[0]]
else:
    ground_truth = ground_truth[:sh_test.shape[0]]

# SNR calulation
SNR = 10 * np.log10(np.sum(ground_truth ** 2) / np.sum((ground_truth - sh_test) ** 2))

print(f"SNR Ratio: {SNR} dB")


# # P1 Audio File_1

# In[49]:


from IPython.display import Audio
P1_audio = 'test_s_01_recons.wav'
Audio(P1_audio)


# In[90]:


# Professor Kim's Voice !?


import librosa
s, sr=librosa.load('train_clean_male.wav', sr=None)
S=librosa.stft(s, n_fft=1024, hop_length=512)
sn, sr=librosa.load('train_dirty_male.wav', sr=None)
X=librosa.stft(sn, n_fft=1024, hop_length=512)

# Apply abs
abs_s = np.abs(S)
abs_x = np.abs(X)

# Moedel experiments, the values of last layer should not be a negative 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(513,)),
    #tf.keras.layers.Dense(256, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='elu'),
    #tf.keras.layers.Dropout(0.5), 
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(513, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dense(513, activation='elu'),
])

model.compile(optimizer = 'adam', loss='mean_squared_error')

model.fit(abs_x.T, abs_s.T, epochs=40, batch_size=128)

# Load test data and apply STFT
test_noisy, sr = librosa.load('test_x_02.wav', sr=None)
test_x_k = librosa.stft(test_noisy, n_fft=1024, hop_length=512)

# Predict test x
test_abs_x_k = np.abs(test_x_k)
p_abs_s_k = model.predict(test_abs_x_k.T).T

#Recover the (complex-valued) speech spectrogram of the test signal
p_hat_s_k = p_abs_s_k * (test_x_k / (test_abs_x_k)) # + 1e-20

# Reconstructin test file
sh_test_k = librosa.istft(p_hat_s_k, hop_length=512)
sf.write('test_s_02_recons.wav', sh_test_k, sr)


# # P1 Audio File_2_Professor Kim's voice

# In[51]:


from IPython.display import Audio
P1_audio_Kim = 'test_s_02_recons.wav'
Audio(P1_audio_Kim)


# In[96]:


pip install --upgrade soundfile


# ## Speech Enhancement using 1D CNN

# This problem was the most difficult problem because the desired result was not easily obtained. I somehow solved this problem, but I think I need to practice a bit more and become familiar with the correlation between filters and kernels and strides and pooling.

# In[47]:


# import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import he_normal

# Data load
s, sr = librosa.load('train_clean_male.wav', sr=None)
sn, sr = librosa.load('train_dirty_male.wav', sr=None)

# STFT(Spectrogram) calculation
abs_s = np.abs(librosa.stft(s, n_fft=1024, hop_length=512))
abs_x = np.abs(librosa.stft(sn, n_fft=1024, hop_length=512))

# Transopose for CNN
abs_s = abs_s.T
abs_x = abs_x.T

# 1D CNN Model 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(513,)), 
    tf.keras.layers.Reshape(target_shape=(513, 1)), 
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='elu'),#strides =2),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='elu', strides =2),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    #tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu',strides =2),
    #tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(256, activation='elu', kernel_initializer=he_normal()),
    tf.keras.layers.Dense(513, activation='relu', kernel_initializer=he_normal())
])

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Traning X and S
model.fit(abs_x, abs_s, epochs=60, batch_size=256)

# Load test data and apply STFT
test_noisy, sr = librosa.load('test_x_01.wav', sr=None)
test_x = librosa.stft(test_noisy, n_fft=1024, hop_length=512).T
test_x_abs = np.abs(test_x)

# Predict test x
p_abs_s_cnn = model.predict(test_x_abs)

# Recover the (complex-valued) speech spectrogram of the test signal
p_hat_s_cnn = p_abs_s_cnn * (test_x / (test_x_abs + 1e-20))
sh_test_cnn = librosa.istft(p_hat_s_cnn.T, hop_length=512)
sf.write('test_s_01_recons_cnn.wav', sh_test_cnn, sr)

# Compare with ground truth
ground_truth, sr = librosa.load('test_s_01.wav', sr=None)

# Adjust length between two files
if sh_test_cnn.shape[0] > ground_truth.shape[0]:
    sh_test_cnn = sh_test_cnn[:ground_truth.shape[0]]
else:
    ground_truth = ground_truth[:sh_test_cnn.shape[0]]

# SNR calulation
SNR = 10 * np.log10(np.sum(ground_truth ** 2) / np.sum((ground_truth - sh_test_cnn) ** 2))

print(f"SNR Ratio: {SNR} dB")


# # P2 Audio File

# In[52]:


from IPython.display import Audio
P2_audio = 'test_s_01_recons_cnn.wav'
Audio(P2_audio)


# ## CIFAR-10 Classification with Data Augmentation

# I tried applying CNN to image training for the first time through this problem. In particular, I learned how to normalize an image and then rescale it to a range between -1 and 1. I also learned what role augmented data plays.
# ### I wanted to check the accuracy for every epoch, so I didn't set verbose = 0. I apologize for the long scroll, and I appreciate your patience. 

# In[23]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import he_normal
import random
from keras.datasets import cifar10


# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Set X_train and x_val
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

#  CNN classifier
# Reshape images and normalize
x_train = x_train.reshape(32, 32, 3) / 255.0
x_val = x_val.reshape(32, 32, 3) / 255.0
x_train = (x_train - 0.5) * 2
x_val = (x_val - 0.5) * 2

# Build CNN model with He initializer
model = models.Sequential()
model.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer=he_normal()))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Conv2D(10, (5, 5), activation='relu', kernel_initializer=he_normal()))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(20, activation='relu', kernel_initializer=he_normal()))
model.add(layers.Dense(10, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train and record validation accuracy over epochs
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=128)

# Returen to the Normal [0,1]

# ver.1
#x_train = (x_train / 2) + 0.5
#x_val = (x_val / 2) + 0.5

#ver.2
x_train = (x_train + 1) / 2
x_val = (x_val + 1) / 2

# Build another classifier using augmented dataset


# Brighten images
brightened_x_train = np.minimum(1.1 * x_train, 1) # more than 1 

# Darken images
darkened_x_train = 0.9 * x_train # less than 1

# Flip images horizontally
flipped_x_train = np.flip(x_train, axis=2)

# Merge datasets
augmented_x_train = np.vstack([brightened_x_train, darkened_x_train, flipped_x_train, x_train])
augmented_y_train = np.vstack([y_train, y_train, y_train, y_train])
print(augmented_x_train.shape)

# Visualize augmented images
num_samples = 10

fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
for i in range(num_samples):
    index = random.randint(0, len(augmented_x_train) - 1)
    axs[i].imshow(augmented_x_train[index]) 
    axs[i].set_title(f"Sample {index}")
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# Scale back to [-1,1]
augmented_x_train = (augmented_x_train - 0.5) * 2
x_val = (x_val - 0.5) * 2

# Train with the augmented dataset
model_augmented = models.Sequential()
model_augmented.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer=he_normal()))
model_augmented.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model_augmented.add(layers.Conv2D(10, (5, 5), activation='relu', kernel_initializer=he_normal()))
model_augmented.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model_augmented.add(layers.Flatten())
model_augmented.add(layers.Dense(20, activation='relu', kernel_initializer=he_normal()))
model_augmented.add(layers.Dense(10, activation='softmax'))

model_augmented.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_augmented = model_augmented.fit(augmented_x_train, augmented_y_train, validation_data=(x_val, y_val), epochs=200, batch_size=128)

# Visualize validation accuracy
plt.figure(figsize=(12, 5))
plt.plot(history.history['val_accuracy'], label='Original')
plt.plot(history_augmented.history['val_accuracy'], label='Augmented')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ## Self-Supervised Pretext Task + Transfer Learning (CIFAR-10)

# The challenging aspect of this problem was setting up the transfer model. In the part where I retrieve the weights, I applied the weights, removed the last layer using pop(), and then reinserted the softmax layer with an output value of 10. The next difficult part was applying different learning rates to each layer, and I solved this using the 'tfa.optimizers.MultiOptimizer' method. However, the most demanding aspect of this  problem was that there were so many epochs that it took a lot of time. 
# ### I wanted to check the accuracy for every epoch, so I didn't set verbose = 0. I apologize for the long scroll, and I appreciate your patience. To see the final results, just scroll to the very end.

# In[21]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Set aside 500 examples for baseline and transfer model
baseline_x_train = x_train[-500:]
baseline_y_train = y_train[-500:]
x_train = x_train[:-500]
y_train = y_train[:-500]

# Prepare augmented data for pretext model
# Class 0: Original
# Class 1: Flip in a vertical way
# Class 2: Rotate 90 degree counter clock wise
X_class_0 = x_train
X_class_1 = np.flip(x_train, axis=1)
X_class_2 = np.rot90(x_train, k=1, axes=(1, 2))

Y_class_0 = np.zeros(y_train.shape)
Y_class_1 = np.ones(y_train.shape)
Y_class_2 = 2 * np.ones(y_train.shape)
                     
pretext_x_train = np.vstack([X_class_0 , X_class_1, X_class_2])
pretext_y_train = np.vstack([Y_class_0, Y_class_1, Y_class_2])

#Normalize
pretext_x_train  = pretext_x_train.reshape(-1, 32, 32, 3) / 255.0
pretext_x_train  = (pretext_x_train  - 0.5) * 2

baseline_x_train = baseline_x_train.reshape(-1, 32, 32, 3) / 255.0
baseline_x_train = (baseline_x_train  - 0.5) * 2

x_test = x_test.reshape(-1, 32, 32, 3) / 255.0
x_test = (x_test - 0.5) * 2

print(pretext_x_train.shape)
print(pretext_y_train.shape)





# Pretext model (Problem 3 model)
model_pretext = models.Sequential()
model_pretext.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer='he_normal'))
model_pretext.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model_pretext.add(layers.Conv2D(10, (5, 5), activation='relu', kernel_initializer='he_normal'))
model_pretext.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model_pretext.add(layers.Flatten())
model_pretext.add(layers.Dense(20, activation='relu', kernel_initializer='he_normal'))
model_pretext.add(layers.Dense(3, activation='softmax'))  # 3 Classes

# Compile and Fit
model_pretext.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_pretext = model_pretext.fit(pretext_x_train, pretext_y_train, epochs=200, batch_size=256)

# Save for transfer model
model_pretext.save("pretext_param.h5")





# Baseline Model
model_baseline = models.Sequential()
model_baseline.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer='he_normal'))
model_baseline.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model_baseline.add(layers.Conv2D(10, (5, 5), activation='relu', kernel_initializer='he_normal'))
model_baseline.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model_baseline.add(layers.Flatten())
model_baseline.add(layers.Dense(20, activation='relu', kernel_initializer='he_normal'))
model_baseline.add(layers.Dense(10, activation='softmax'))  # 10 classes

# Compile and fit

model_baseline.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_baseline = model_baseline.fit(baseline_x_train, baseline_y_train, validation_data=(x_test, y_test), 
                                      epochs=10000, batch_size=500)





# Transfer Learning model
model_transfer = models.Sequential()
model_transfer.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(32, 32, 3)))
model_transfer.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model_transfer.add(layers.Conv2D(10, (5, 5), activation='relu'))
model_transfer.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model_transfer.add(layers.Flatten())
model_transfer.add(layers.Dense(20, activation='relu'))
model_transfer.add(layers.Dense(3, activation='softmax'))  # 3 Classes

# Load and set pretext weights 
model_transfer.load_weights("pretext_param.h5")
model_transfer.set_weights(model_pretext.get_weights())

# Delete and insert last layer for 10 classes and weights with He_normal                 
model_transfer.pop()
model_transfer.add(layers.Dense(10, activation='softmax', kernel_initializer='he_normal'))

# Learning rate setting: last =1e-3 / others = 1e-5
optimizers = [
    tf.keras.optimizers.Adam(learning_rate=1e-5),
    tf.keras.optimizers.Adam(learning_rate=1e-3)
]
optimizers_and_layers = [(optimizers[0], model_transfer.layers[:-1]), (optimizers[1], model_transfer.layers[-1])]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

#Compile and fit
model_transfer.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_transfer = model_transfer.fit(baseline_x_train, baseline_y_train,validation_data=(x_test, y_test),epochs=10000,
                                     batch_size=500)   
 

    
    
    
    
# Extract validation accuracy for every 100th epoch
baseline_val_acc = history_baseline.history['val_accuracy'][::100]
transfer_val_acc = history_transfer.history['val_accuracy'][::100]

# Create a list of epochs for the x-axis (every 100th epoch)
epochs = list(range(0, len(baseline_val_acc) * 100, 100))

# Plot the validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, baseline_val_acc, label='Baseline Model')
plt.plot(epochs, transfer_val_acc, label='Transfer Model')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison (Baseline vs. Transfer)')
plt.legend()
plt.grid(True)
plt.show()    

