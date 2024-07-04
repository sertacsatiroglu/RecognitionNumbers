#Importing required libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical, plot_model

import matplotlib.pyplot as plt
import numpy as np

#Ignoring not required warnings to preventing unexpected situations
import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)
filterwarnings('ignore')

#Uploading mnist dataset for numbers in range of [0,9]
(x_train, y_train), (x_test,y_test) = mnist.load_data()

#observing size of dataset
print("size of train set : ", x_train.shape, y_train.shape)
print("size of test set : ", x_test.shape, y_test.shape)

#Determining number of unique labels, namely how many it is how many different label we have.
num_labels = len(np.unique(y_train))

#Showing random instances from data-set
plt.figure(figsize = (10,10))
plt.imshow(x_train[5900], cmap = 'gray')

#After this point, we will examine the images by using its RGB values in its pixels.

#We will define a function to observe random 10 instances from data-set
def visualize_img(data):
  plt.figure(figsize = (10,10))
  for n in range(10):
    ax = plt.subplot(5,5,n+1)
    plt.imshow(x_train[n], cmap = 'gray')
    plt.axis('off')

#We will determine size of piece of data for further processes
x_train[2].shape
#Also the main idea is observing rgb values of common labels and determine which label it is for prediction, or you can use sum for same purpose.
x_train[2].mean()
x_train[2].sum()

#Visualizing the images.
def pixel_visualize(img):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    threshold = img.max() / 2.5

    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y,x),
                        color='white' if img[x][y]<threshold else 'black')

#Testing visualized pixel.
pixel_visualize(x_train[2])

#Encoding process e.g. [0 1 2 3 4 ==  0 1 0 0 0] for number 1.
#Before encoding.
y_train[0:5]
#After encoding.
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)

#Reshaping process
image_size = x_train.shape[1]
image_size

#We add one information element in size array of pixels.
print(f"x_train boyutu: {x_train.shape}")
print(f"x_test boyutu: {x_test.shape}")

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(f"x_train boyutu: {x_train.shape}")
print(f"x_test boyutu: {x_test.shape}")

#Normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#Modeling and checking summary of modeling
model = tf.keras.Sequential([
    Flatten(input_shape = (28,28,1)),
    Dense(units=128, activation='relu', name='layer1'),
    Dense(units=num_labels, activation='softmax', name='output_layer')])

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),"accuracy"])
model.summary()

#Fit function to train the model for given data-set
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test,y_test))

#Saving the results to history.
history = model. fit (x_train, y_train, epochs=10, batch_size=128, validation_data= (x_test, y_test))

#Accuracy graph
plt.figure(figsize= (20, 5))
plt.subplot (1, 2, 1)
plt.plot (history. history[ 'accuracy'], color='b', label='Training Accuracy')
plt.plot (history. history[ 'val_accuracy'], color='r', label='Validation Accuracy')
plt.legend (loc='lower right')
plt.xlabel ( 'Epoch', fontsize=16)
plt.ylabel ('Accuracy', fontsize=16)
plt.ylim([min(plt.ylim()),1])
plt.title('Egitim ve Test Bagarim Grafigi', fontsize=16)

#Loss Graph
plt.subplot (1, 2, 2)
plt.plot (history. history[ 'loss'], color='b', label='Training Loss')
plt.plot (history. history['val_loss'], color='r', label='Validation Loss')
plt. legend (loc= 'upper right')
plt.xlabel ( 'Epoch', fontsize=16)
plt.ylabel( 'Loss', fontsize=16)
plt.ylim([0,max(plt.ylim() )])
plt.title('Egitim ve Test Kayip Grafigi', fontsize=16)

#Checking the values of Accuracy,Loss, Precision, Recall
loss, precision, recall, acc = model.evaluate (x_test, y_test, verbose=False)
print ("\nTest Accuracy: %.1f%%" % (100.0 * acc))
print ("\nTest Loss: %.1f%%" % (100.0 * loss))
print ("\nTest Precision: %.1f%%" % (100.0 * precision))
print ("\nTest Recall: %.1f%%" % (100.0 * recall))

#Saving the model
model.save('mnist_model.h5')

#Prediction Part of a Random Data
import random
random = random.randint(0,x_test.shape[0])
test_image = x_test[random]
y_test[random]
plt.imshow(test_image.reshape(28,28), cmap ='gray')
test_data = x_test[random].reshape(1,28,28,1)
probability = model.predict(test_data)
predicted_classes = np.argmax(probability)

#Checking the result of prediction.
predicted_classes
