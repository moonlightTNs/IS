import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import os
from models.Machine_model import (
    load_classification_data,
    train_classification_models,
    load_regression_data,
    train_regression_model
)

# Sidebar for navigation
st.sidebar.title("📌Menu")
page = st.sidebar.radio("🔍 Select menu", ["📊Classification", "📈Regression"])

#--------------------------------------------------------------------------------------------------------------

if page == "📊Classification":
    st.title("Classification Model - iris-flower")
    file_path = "is/datasets/IRIS.csv"
    
    try:
        df = pd.read_csv(file_path)
        st.success("Iris dataset loaded successfully!")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

   
    X, y, encoder = load_classification_data(file_path)

    st.write("### Dataset Information")
    st.write("The Iris dataset contains 150 samples of iris flowers, with 50 samples from each of three species: Iris setosa, Iris versicolor, and Iris virginica. Each sample has four features:")
    st.write("- **Sepal Length**: The length of the sepal in centimeters.")
    st.write("- **Sepal Width**: The width of the sepal in centimeters.")
    st.write("- **Petal Length**: The length of the petal in centimeters.")
    st.write("- **Petal Width**: The width of the petal in centimeters.")
    st.write("The target variable is the species of the iris flower, which is a categorical variable with three possible values: setosa, versicolor, and virginica.")

    st.write("### Full Dataset:")
    st.dataframe(df)
     # Display missing values before data cleansing
    st.write("### Missing Values Before Data Cleansing")
    st.write(df.isnull().sum())

    # Data Cleansing (example: dropping rows with missing values)
    df.dropna(inplace=True)

    # Display missing values after data cleansing
    st.write("### Missing Values After Data Cleansing")
    st.write(df.isnull().sum())


    st.write("### Train Models")
    rf_model, svm_model, X_test, y_test = train_classification_models(X, y)

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
    file_path = "datasets/diabetes_prediction_dataset.csv"
    
    try:
        df = pd.read_csv(file_path)
        st.success("is/Diabetes dataset loaded successfully!")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    

    X, y = load_regression_data(file_path)

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

    st.write("### Full Dataset:")
    st.dataframe(df)
    
    # Display missing values before data cleansing
    st.write("### Missing Values Before Data Cleansing")
    st.write(df.isnull().sum())

    # Data Cleansing (example: filling missing values with mean)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Display missing values after data cleansing
    st.write("### Missing Values After Data Cleansing")
    st.write(df.isnull().sum())

    st.write("### Train Model")
    reg_model, X_test, y_test = train_regression_model(X, y)
    y_pred = reg_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"` Mean Squared Error (MSE): {mse:.2f}`")
    st.write(f"` R² Score: {r2:.2f}`")

    st.write("### Predicted vs Actual Values")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    st.pyplot(fig)

    st.write("### Residual Plot")
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5, color='purple')
    ax.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linestyles='dashed', colors='red')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    st.pyplot(fig)

    st.write("### Distribution of Residuals")
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax, color='green')
    plt.xlabel("Residuals")
    plt.title("Distribution of Residuals")
    st.pyplot(fig)

    # Reasons for using the algorithms
    st.write("### Reasons for Using the Algorithms")
    st.write("- **Linear Regression**: It is a simple and interpretable model that works well when there is a linear relationship between the features and the target variable. It is useful for understanding the impact of each feature on the target variable.")
    st.write("- **Random Forest Regressor**: It is an ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting. It works well with complex datasets and can capture non-linear relationships between the features and the target variable.")



