import streamlit as st

st.title("Neural Network Application")

st.write("""
    
## 1️⃣ Data Preparation

### 📂 Loading Data
- Used `TensorFlow` to load data from the `CIFAR-10 dataset`.
- Loaded dataset using `datasets.cifar10.load_data()` from TensorFlow.
- Inspected the dataset to understand the structure and types of data.

### 🎯 Filtering Selected Classes
- **selected_classes = [3, 5, 8]** This code selects only three classes: `cat (3)` , `dog (5)` , and `ship (8)`.
- `train_filter = np.isin(train_labels, selected_classes).flatten()`
- Uses  `np.isin()` to filter out unwanted categories, keeping only images of cats, dogs, and ships.

### 🎨 Normalizing Pixel Values
- `train_images, test_images = train_images / 255.0, test_images / 255.0`
- Normalized pixel values to be between `0 and 1` for better model performance.

### 🔢 Mapping Labels to New Indexes (0, 1, 2)
- `label_map = {3: 0, 5: 1, 8: 2}`
- `train_labels = np.vectorize(label_map.get)(train_labels)`
- `test_labels = np.vectorize(label_map.get)(test_labels)`
- Since only three categories are used, the labels are remapped for correct classification indexing.

---

## 2️⃣ Theory of Algorithms Used
##### This model is a `Convolutional Neural Network (CNN)` designed to process image data effectively.                                 
### 🏗️ Conv2D Layer
- `layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),` 
- Uses 32 filters of size 3x3 to extract spatial features from images.

### 🏊 MaxPooling Layer
- `layers.MaxPooling2D((2, 2)),`
- Downsamples the feature maps, reducing dimensions while retaining important features.

### 🔄 Flatten Layer
- `layers.Flatten(),`
- Converts the 2D feature maps into a 1D array for the fully connected layers.

### 🎛️ Fully Connected Layers
- `layers.Dense(64, activation='relu'),`
- `layers.Dense(3, activation='softmax')`
- The final layer uses softmax activation to classify the image into one of three categories.

---

## 3️⃣ Model Development Steps

### ⚙️ Compiling the Model
- Used `model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])`.
- Uses Adam optimizer and sparse_categorical_crossentropy loss function for multi-class classification.

### 🏋️ Training the Model
- Used `model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))`.
- Trained the model for 10 epochs on the training data with validation on the test data.

### 💾 Saving Model in session_state
- `st.session_state['model_trained'] = True`
- `st.session_state['history'] = history`
- `st.session_state['model'] = model`
- Saved the trained model and training history in the session state for future use.
- Ensures the model is only trained once when running the Streamlit app.

---
## 4️⃣ Model Evaluation

### 📈 Plotting Accuracy and Loss Graphs
- Used `st.line_chart(history.history['accuracy'])` to plot the accuracy graph.
- Used `st.line_chart(history.history['loss'])` to plot the loss graph.
- Visualized the training and validation metrics to evaluate model performance.

### 🔢 Computing Confusion Matrix
- Used `confusion_matrix(test_labels, y_pred)` to compute the confusion matrix.
- Helps analyze misclassification patterns.

### 🎯 Calculating Accuracy
- `accuracy = np.sum(y_pred == test_labels.flatten()) / len(test_labels)`
- Calculated the accuracy of the model on the test data.

---
### Reference🔗
#### Dataset : 
- Use [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
#### TensorFlow Documentation
- Use [TensorFlow](https://www.tensorflow.org/)

""")
