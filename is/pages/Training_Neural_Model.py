import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# Sidebar for navigation
st.sidebar.title("📌Menu")
page = st.sidebar.radio("🔍 Select menu", ["🧠📸CNN", "🏗️🤖Model"])

# Train the model only once
if 'model_trained' not in st.session_state:
    with st.spinner('Training the model...'):
        history = model.fit(train_images, train_labels, epochs=10, 
                            validation_data=(test_images, test_labels))
    st.session_state['model_trained'] = True
    st.session_state['history'] = history
    st.session_state['model'] = model

if page == "🧠📸CNN":
    st.title("Convolutional Neural Network (CNN) for CIFAR-10")
    st.write("### Example Dataset")
    # Display example images from the dataset
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(test_images[i])
        ax.set_title(class_names[test_labels[i][0]])
        ax.axis('off')
    st.pyplot(fig)

    st.write("### Training and Validation Metrics")
    history = st.session_state['history']

    # Plot training & validation accuracy values
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train')
    ax.plot(history.history['val_accuracy'], label='Test')
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper left')
    st.pyplot(fig)  # Display the plot in Streamlit

    # Plot training & validation loss values
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Test')
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper left')
    st.pyplot(fig)  # Display the plot in Streamlit

    # Calculate and display confusion matrix
    if 'model' in st.session_state:
        model = st.session_state['model']
        y_pred = np.argmax(model.predict(test_images), axis=1)
        cm = confusion_matrix(test_labels, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        # Calculate and display accuracy
        accuracy = np.sum(y_pred == test_labels.flatten()) / len(test_labels)
        st.write(f"#### Model accuracy: `{accuracy * 100:.2f}%`")

if page == "🏗️🤖Model":
    st.title("Predict with the Model")
    st.write("You can upload an image `cat`, `dog`, `ship` to make a prediction.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)
        
        # Preprocess the image
        image_resized = image.resize((32, 32))
        image_array = np.array(image_resized)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make a prediction
        if 'model' in st.session_state:
            model = st.session_state['model']
            prediction = model.predict(image_array)
            class_name = class_names[np.argmax(prediction)]
            st.write(f"#### The model predicts this image is a: `{class_name}`")
        
        # Display the image
        st.image(image, caption='Uploaded Image.', use_container_width=True, width=300)  # Limit the width to 300 pixels