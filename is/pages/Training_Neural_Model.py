import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.CNN_model import get_model, get_class_names, get_test_data

# Sidebar for navigation
st.sidebar.title("📌Menu")
page = st.sidebar.radio("🔍 Select menu", ["🧠📸CNN", "🏗️🤖Model"])

# Check if the model is already trained and stored in session state
if 'model_trained' not in st.session_state:
    with st.spinner('Loading the model...'):
        model = get_model()
        train_images, train_labels, test_images, test_labels = get_test_data()
        class_names = get_class_names()
    with st.spinner('Training the model...'):
        history = model.fit(train_images, train_labels, epochs=10, 
                            validation_data=(test_images, test_labels))
    st.session_state['model_trained'] = True
    st.session_state['history'] = history
    st.session_state['model'] = model
    st.session_state['class_names'] = class_names
else:
    model = st.session_state['model']
    history = st.session_state['history']
    if 'class_names' not in st.session_state:
        st.session_state['class_names'] = get_class_names()
    class_names = st.session_state['class_names']
    train_images, train_labels, test_images, test_labels = get_test_data()

if page == "🧠📸CNN":
    st.title("Convolutional Neural Network (CNN) for CIFAR-10")
    st.write("### Example Dataset")
    # Display example images from the dataset
    fig, axes = plt.subplots(1, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        idx = np.where(test_labels == i)[0][0]
        ax.imshow(test_images[idx])
        ax.set_title(class_names[i])
        ax.axis('off')
    st.pyplot(fig)

    st.write("### Training and Validation Metrics")

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
    st.write("You can upload images `airplane` , `automobile`, `bird`, `cat`, `deer` to make predictions.")
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
        prediction = model.predict(image_array)
        class_name = class_names[np.argmax(prediction)]
        st.write(f"#### The model predicts this image is a: `{class_name}`")
        
        # Display the image
        st.image(image, caption='Uploaded Image.', use_container_width=True, width=300)  # Limit the width to 300 pixels