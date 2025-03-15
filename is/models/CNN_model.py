import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Define the class names
class_names = ['cat', 'dog', 'ship']

# Filter the dataset to include only the selected classes
selected_classes = [3, 5, 8]  # cat, dog, ship
train_filter = np.isin(train_labels, selected_classes).flatten()
test_filter = np.isin(test_labels, selected_classes).flatten()

train_images, train_labels = train_images[train_filter], train_labels[train_filter]
test_images, test_labels = test_images[test_filter], test_labels[test_filter]

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Map the labels to the new class indices
label_map = {3: 0, 5: 1, 8: 2}
train_labels = np.vectorize(label_map.get)(train_labels)
test_labels = np.vectorize(label_map.get)(test_labels)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # Change to 3 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Save the model and history in session state
def get_model():
    return model

def get_history():
    return history

def get_class_names():
    return class_names

def get_test_data():
    return train_images, train_labels, test_images, test_labels