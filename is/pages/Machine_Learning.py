import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Title
st.title("Iris Flower Classification")

# Load dataset
file_path = "dataset/IRIS.csv"
# ตรวจสอบว่าไฟล์มีอยู่จริง
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.success("Dataset loaded successfully!")
else:
    st.error("File not found. Please upload the dataset.")



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
    
    
    if 'df' in locals():
        st.write("### Preview of Dataset (Before Cleaning)")
        st.dataframe(df.head())

    # Data Cleaning
    st.write("## Data Cleaning")

    # ตรวจสอบค่า Missing
    st.write("### 🔍 Missing Values (Before Cleaning)")
    st.write(df.isnull().sum())

    

    # คอลัมน์ตัวเลขที่ต้องใช้
    num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # แปลงให้เป็นตัวเลข และจัดการค่า NaN
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # เติมค่า NaN ด้วยค่าเฉลี่ยของคอลัมน์นั้น
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # ตรวจสอบ species ที่มีค่าผิดปกติ และแก้ไข
    df["species"] = df["species"].astype(str).str.extract(r'(Iris-setosa|Iris-versicolor|Iris-virginica)', expand=False)

    # ลบแถวที่มีค่า NaN หลังจากการแก้ไขข้อมูลผิดพลาด
    df.dropna(inplace=True)


    # ตรวจสอบค่า Missing หลังทำความสะอาด
    st.write("### 🔍 Missing Values (After Cleaning)")
    st.write(df.isnull().sum())
    
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
