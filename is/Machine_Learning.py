import streamlit as st
import asyncio


# ตรวจสอบ event loop
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


st.title("Machine Learning Application")

st.write("""
    
## 1️⃣ Data Preparation

### 🔄 Loading Data
- Used `pandas` to read data from the file `IRIS.csv | diabetes_prediction_dataset.csv`.
- Inspected the dataset to understand the structure and types of data.

### 🧹 Handling Missing Values
- Used `fillna()` method to replace missing values with the mean of each column.
- Used `dropna()` to remove rows with missing values.
- Ensures data completeness and reduces bias caused by missing values.
- Checked for outliers using **Boxplot** and handled extreme values.

### 🔍 Separating Features and Labels
##### Data Preprocessing
- **X (Features):):** Input variables used for prediction, such as information about flower characteristics or health data.
- **y (Label):** Output variable to predict, such as the species of the flower or diabetes status.
- Check the relationship of Features to eliminate redundant variables.

### 🔄 Train-Test Split
- Used `train_test_split()` to split the dataset (80% training, 20% testing).

### ⚖️ Feature Scaling
- Used `StandardScaler()` to normalize feature values.
- Applied `fit_transform()` on training data and `transform()` on test data.
- Helps improve model performance by standardizing feature ranges.

---

## 2️⃣ Theory of Algorithms Used

### 🌳 1. Random Forest Classifier
- **Random Forest** uses an ensemble of Decision Trees for robust classification.
- It helps reduce overfitting and performs well on datasets with multiple important features.
- **Weakness:** It can be slow when working with large datasets.

### 🧠 2. Support Vector Machine (SVM)
- **SVM** finds the optimal hyperplane for classification.
- **Weakness:** It is computationally expensive for large datasets.

---

## 3️⃣ Model Development Steps

### ✅ Training the Model
- Used `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Trained using `rf_model.fit(X_train, y_train)`.

### 🔮 Making Predictions
- Used `predict()` to generate predictions on the test data.
- Stored predictions for evaluation and analysis.

### 📊 Evaluating the Model
- Used `accuracy_score` to evaluate model performance.
- Generated a **classification_report** to assess `Precision`, `Recall`, and`F1-score`.
- Used **Confusion Matrix** for better insights into classification errors.

---
### Reference🔗
#### Dataset : 
- Classification : Use [IRIS](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- Regression : Use [diabetes_prediction_dataset](https://www.kaggle.com/datasets/dat00700/diabetes-prediction-dataset)
#### Scikit-learn Documentation
- Use [Scikit-learn](https://scikit-learn.org/stable/index.html)
""")
