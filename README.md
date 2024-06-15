# Object_Detection43
# Problem Statement:
Build a mobile-friendly neural network model using TensorFlow Lite to recognize and classify handwritten digits.

# Background:
Handwritten digit recognition is a classic problem in machine learning and deep learning. The MNIST dataset, which contains images of handwritten digits from 0 to 9, is widely used for training and testing models for this task. With the rise of mobile devices, there's a need for lightweight models that can run on a device with minimal computational resources. Using TensorFlow Lite, provide a solution for this by allowing the conversion of TensorFlow models to a format optimized for on-device applications.

# Tasks:

Import necessary libraries.
Data Loading and Preprocessing:
Load the MNIST dataset.
Normalize the images so that pixel values are between 0 and 1.
Model Building and Training:
Define a neural network model using TensorFlow/Keras.
Compile the model, specifying an optimizer, loss function, and metrics.
Conversion to TensorFlow Lite:
Convert the trained TensorFlow model to TensorFlow Lite format.
Save the converted model to a file.
Testing and Visualization:
Load a sample handwritten digit image from the test set.
Preprocess the image to match the input format of the model.
Use the TensorFlow Lite interpreter to predict the digit in the sample image.
Get input and output tensors and set the tensor to the input image.
Invoke the model and obtain the output of the model.
Display the sample image and the model's prediction.
Interpretation:
Compare the model's prediction with the actual label of the sample image to determine the accuracy of the model on this sample.
Provide a brief interpretation of the results, discussing the model's performance and any observed discrepancies.
Import the necessary libraries.
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

#Model Building and Training
#Define a neural network model using TensorFlow/Keras.
#Compile the model, specifying an optimizer, loss function, and metrics
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Convert the trained TensorFlow model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file 
with open('mnist_model.tflite', 'wb') as f:
    f.write(tflite_model)
```
Testing and Visualization
Load a sample handwritten digit image from the test set.
Preprocess the image to match the input format of the model.
Use the TensorFlow Lite interpreter to predict the digit in the sample image.
Get input and output tensors and set the tensor to the input image.
Invoke the model and obtain the output of the model.
Display the sample image and the model's prediction.
```python
import matplotlib.pyplot as plt

def preprocess_image(img):
    '''Preprocess an image to match MNIST format.'''
    return img.reshape(1, 28, 28).astype(np.float32)

# Choose a sample image from the test set
sample_image = test_images[0]
sample_label = test_labels[0]

# Display the sample image
plt.imshow(sample_image, cmap='gray')
plt.title(f"True Label: {sample_label}")
plt.axis('off')  # Hide axes
plt.show()

# Preprocess the image
processed_image = preprocess_image(sample_image)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="mnist_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the tensor to the input image
interpreter.set_tensor(input_details[0]['index'], processed_image)

# Invoke the model
interpreter.invoke()

# Obtain the output of the model
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_digit = np.argmax(output_data)

print(f"Predicted Label: {predicted_digit}")
```
### Interpretation
Compare the model's prediction with the actual label of the sample image to determine the accuracy of the model on this sample.
Provide a brief interpretation of the results, discussing the model's performance and any observed discrepancies.
# Interpretation
Displayed Image: The displayed image represents a handwritten digit from the MNIST test dataset. This image serves as a visual reference for us to see what digit the model is trying to recognize. The title above the image, "True Label", indicates the actual number represented by the handwritten digit, as per the test dataset.

Predicted Label: This is the digit that our TensorFlow Lite model predicts the handwritten digit to be. The model arrives at this prediction based on patterns it learned during training.
