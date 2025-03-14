import streamlit as st

st.title("Machine Learning Application")


st.write("""
    
## 1️⃣ Data Preparation

### 🔹 Loading Data
- Used `pandas` to read data from the file `IRIS.csv | diabetes_prediction_dataset.csv`.
- Load dataset from: [Kaggle](https://www.kaggle.com/).
- Inspected the dataset to understand the structure and types of data.

### 🔹 Handling Missing Values
- Used `fillna()` method to replace missing values with the mean of each column.
- Ensures data completeness and reduces bias caused by missing values.
- Checked for outliers using **Boxplot** and handled extreme values.

### 🔹 Separating Features and Labels
- **X (Features):** pH, Hardness, Solids, Chloramines, Conductivity, etc.
- **y (Label):** Potability (0 = Not Drinkable, 1 = Drinkable).
- Checked feature correlation to remove redundant variables.

### 🔹 Train-Test Split
- Used `train_test_split()` to split the dataset (80% training, 20% testing).
- Applied `stratify=y` to maintain class distribution.
- Implemented `StratifiedKFold()` for balanced data splitting.

### 🔹 Feature Scaling
- Used `StandardScaler()` to normalize feature values.
- Applied `fit_transform()` on training data and `transform()` on test data.
- Helps improve model performance by standardizing feature ranges.

---

## 2️⃣ Theory of Algorithms Used

### 🔹 1. Random Forest Classifier 🌲
- Uses an ensemble of **Decision Trees** for robust classification.
- Reduces overfitting and performs well on datasets with multiple important features.
- **Weakness:** Can be slow with large datasets.

### 🔹 2. Logistic Regression 📊
- Uses a **sigmoid function** for binary classification.
- Helps in understanding **feature importance**.
- **Weakness:** Performs poorly on non-linearly separable data.

### 🔹 3. Support Vector Machine (SVM) 📉
- Finds the **optimal hyperplane** for classification.
- Uses **Kernel Trick** to handle **non-linear data**.
- **Weakness:** Computationally expensive for large datasets.

---

## 3️⃣ Model Development Steps

### ✅ Training the Model
- Used `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Trained using `rf_model.fit(X_train, y_train)`.
- Applied **GridSearchCV** to fine-tune hyperparameters.

### ✅ Making Predictions
- Used `predict()` to generate predictions on test data.
- Stored predictions for evaluation.

### ✅ Evaluating the Model
- Used `accuracy_score` and `classification_report`.
- Evaluated **Precision, Recall, and F1-score**.
- Used **Confusion Matrix** for better insights into classification errors.

### ✅ Comparing Model Performance
- Compared **Random Forest, Logistic Regression, and SVM**.
- Analyzed **ROC-AUC Score** and **Precision-Recall Curve**.
- Determined the best model based on real-world generalization ability.

---
 """)