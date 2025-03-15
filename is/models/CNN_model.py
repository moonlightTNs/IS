import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Define the class names for the selected CIFAR-10 classes
selected_classes = [0, 1, 2, 3, 4]  # airplane, automobile, bird, cat, deer
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer']

# Filter the dataset to include only the selected classes
train_filter = np.isin(train_labels, selected_classes).flatten()
test_filter = np.isin(test_labels, selected_classes).flatten()

train_images, train_labels = train_images[train_filter], train_labels[train_filter]
test_images, test_labels = test_images[test_filter], test_labels[test_filter]

# Update labels to be in the range 0-4
train_labels = np.array([np.where(selected_classes == label)[0][0] for label in train_labels])
test_labels = np.array([np.where(selected_classes == label)[0][0] for label in test_labels])

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')  # Change to 5 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def get_model():
    return model

def get_class_names():
    return class_names

def get_test_data():
    return train_images, train_labels, test_images, test_labels