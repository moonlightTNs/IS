import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import os
import gdown


# Sidebar for navigation
st.sidebar.title("📌Menu")
page = st.sidebar.radio("🔍 Select menu", ["📊Classification", "📈Regression"])

#--------------------------------------------------------------------------------------------------------------

if page == "📊Classification":
    st.title("Classification_iris-flower")
    # Load dataset
    file_path = "is/pages/datasets/IRIS.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.success("Dataset loaded successfully!")
    else:
        st.error("File not found. Please upload the dataset.")

    # Dataset Information
    st.write("### Dataset Information")
    st.write("The Iris dataset contains 150 samples of iris flowers, with 50 samples from each of three species: Iris setosa, Iris versicolor, and Iris virginica. Each sample has four features:")
    st.write("- **Sepal Length**: The length of the sepal in centimeters.")
    st.write("- **Sepal Width**: The width of the sepal in centimeters.")
    st.write("- **Petal Length**: The length of the petal in centimeters.")
    st.write("- **Petal Width**: The width of the petal in centimeters.")
    st.write("The target variable is the species of the iris flower, which is a categorical variable with three possible values: setosa, versicolor, and virginica.")

    # Display the entire dataset
    st.write("### Full Dataset:")
    st.dataframe(df)

    # Data Preprocessing
    encoder = LabelEncoder()
    df["species"] = encoder.fit_transform(df["species"])
    X = df.drop(columns=["species"])
    y = df["species"]

    # Data Cleaning
    st.write("### Data Cleaning")
    # Check for missing values
    st.write("#### Missing Values (Before Cleaning)")
    st.write(df.isnull().sum())

    # Fill missing values for numeric columns with the mean of the respective columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill missing values for non-numeric columns with the mode of the respective columns
    non_numeric_cols = df.select_dtypes(include="object").columns
    for col in non_numeric_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Check for missing values after cleaning
    st.write("#### Missing Values (After Cleaning)")
    st.write(df.isnull().sum())

    # Drop any remaining rows with missing values
    df.dropna(inplace=True)

    # Ensure there are no NaN values in the dataset
    if df.isnull().sum().sum() > 0:
        st.error("There are still missing values in the dataset. Please check the data cleaning steps.")

    # Boxplot for checking outliers
    st.write("### Boxplot for Checking Outliers")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    sns.boxplot(x=df["sepal_length"], ax=ax[0, 0])
    sns.boxplot(x=df["sepal_width"], ax=ax[0, 1])
    sns.boxplot(x=df["petal_length"], ax=ax[1, 0])
    sns.boxplot(x=df["petal_width"], ax=ax[1, 1])
    st.pyplot(fig)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fill missing values in train and test sets
    X_train = pd.DataFrame(X_train).fillna(0)
    X_test = pd.DataFrame(X_test).fillna(0)

    # Train Models
    st.write("### Train Models")

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    st.write(f"` Random Forest Accuracy: {rf_accuracy:.2f}`")
    st.write("#### Random Forest Classification Report:")
    st.text(classification_report(y_test, rf_pred, target_names=encoder.classes_))
    st.write("#### Random Forest Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Random Forest Confusion Matrix")
    st.pyplot(fig)

    # Support Vector Machine
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    st.write(f"` SVM Accuracy: {svm_accuracy:.2f}`")
    st.write("#### SVM Classification Report:")
    st.text(classification_report(y_test, svm_pred, target_names=encoder.classes_))
    st.write("#### SVM Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("SVM Confusion Matrix")
    st.pyplot(fig)

    # Reasons for using the algorithms
    st.write("### Reasons for Using the Algorithms")
    st.write("- **Random Forest**: It is an ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting. It works well with the Iris dataset because it can handle the multi-class classification problem effectively and is robust to overfitting.")
    st.write("- **Support Vector Machine**: It is effective in high-dimensional spaces and is versatile with different kernel functions. SVM works well with the Iris dataset because it can find the optimal hyperplane that separates the classes with maximum margin, making it suitable for the multi-class classification problem.")

#--------------------------------------------------------------------------------------------------------------

if page == "📈Regression":
    st.title("Regression Model - Diabetes Prediction")
    
    # Load Regression Dataset
    file_path = "diabetes_prediction_dataset.csv"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            df = pd.read_csv(f)
        st.success("Diabetes dataset loaded successfully!")
    else:
        st.error("File not found. Please upload the dataset.")
    
    # Dataset Information
    st.write("### Dataset Information")
    st.write("The Diabetes dataset contains various features related to diabetes prediction. Each sample has the following features:")
    st.write("- **Gender**: The gender of the individual.")
    st.write("- **Age**: The age of the individual.")
    st.write("- **Hypertension**: Whether the individual has hypertension (1) or not (0).")
    st.write("- **Heart Disease**: Whether the individual has heart disease (1) or not (0).")
    st.write("- **Smoking History**: The smoking history of the individual.")
    st.write("- **BMI**: The body mass index of the individual.")
    st.write("- **HbA1c Level**: The HbA1c level of the individual.")
    st.write("- **Blood Glucose Level**: The blood glucose level of the individual.")
    st.write("- **Diabetes**: The target variable indicating whether the individual has diabetes (1) or not (0).")
    
    # Display the entire dataset
    st.write("### Full Dataset:")
    st.dataframe(df)
    
    # Data Cleansing
    st.write("### Data Cleansing")
    # Check for missing values
    st.write("#### Missing Values (Before Cleaning)")
    st.write(df.isnull().sum())
    
    # Fill missing values for numeric columns with the mean of the respective columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Fill missing values for non-numeric columns with the mode of the respective columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    for col in non_numeric_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    
    # Check for missing values after cleaning
    st.write("#### Missing Values (After Cleaning)")
    st.write(df.isnull().sum())
    
    # Drop any remaining rows with missing values
    df.dropna(inplace=True)
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Standardize features
    scaler = StandardScaler()
    X = df.drop(columns=["diabetes"])  # Features
    y = df["diabetes"]  # Target Variable
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Regression Model (Linear Regression)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)
    
    # Boxplot for checking outliers
    st.write("### Boxplot for Checking Outliers")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    sns.boxplot(x=df["age"], ax=ax[0, 0])
    sns.boxplot(x=df["bmi"], ax=ax[0, 1])
    sns.boxplot(x=df["HbA1c_level"], ax=ax[1, 0])
    sns.boxplot(x=df["blood_glucose_level"], ax=ax[1, 1])
    st.pyplot(fig)
    
    # Evaluate Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"` Mean Squared Error (MSE): {mse:.2f}`")
    st.write(f"` R² Score: {r2:.2f}`")
    
    # Scatter Plot
    st.write("### Predicted vs Actual Values")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    st.pyplot(fig)
    
    # Residual Plot
    st.write("### Residual Plot")
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5, color='purple')
    ax.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linestyles='dashed', colors='red')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    st.pyplot(fig)
    
    # Distribution Plot of Residuals
    st.write("### Distribution of Residuals")
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax, color='green')
    plt.xlabel("Residuals")
    plt.title("Distribution of Residuals")
    st.pyplot(fig)
    
    st.write("#### Algorithm Used: **Linear Regression**")
    st.write("Linear Regression is used because it is a simple yet effective algorithm for predicting continuous numerical values.")



