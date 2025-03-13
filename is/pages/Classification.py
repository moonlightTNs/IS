import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Title
st.title("Iris Flower Classification")

# Load dataset
file_path = "dataset/IRIS.csv"
df = pd.read_csv(file_path)

# Data Cleaning
st.write("### Data Cleaning")
# Check for missing values
st.write("#### Missing Values:")
st.write(df.isnull().sum())

# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Check for incorrect data types
st.write("#### Data Types:")
st.write(df.dtypes)

# Convert columns to appropriate data types if necessary
# Example: df['column_name'] = df['column_name'].astype('float')

# Top Menu Navigation
menu = st.tabs(["Preview","Dataset", "Evaluation"])

with menu[0]:
    st.write("### Preview of Dataset:")
    st.dataframe(df.head())
    st.write("### Summary Statistics:")
    st.write(df.describe())
    st.write("### Data Distribution:")
    st.bar_chart(df["species"].value_counts())
    
with menu[1]:
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
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.write("### Model Training Complete!")
    st.success("Model is ready for evaluation.")


with menu[2]:
    # Predict
    y_pred = model.predict(X_test)
    
    # Accuracy & Report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    st.write("### Classification Report:")
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Run with: streamlit run script.py
