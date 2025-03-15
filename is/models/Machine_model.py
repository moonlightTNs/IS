import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import os

def load_classification_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        encoder = LabelEncoder()
        df["species"] = encoder.fit_transform(df["species"])
        X = df.drop(columns=["species"])
        y = df["species"]
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        return X, y, encoder
    else:
        raise FileNotFoundError("File not found. Please upload the dataset.")

def train_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    return rf_model, svm_model, X_test, y_test

def load_regression_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.get_dummies(df, drop_first=True)
        scaler = StandardScaler()
        X = df.drop(columns=["diabetes"])
        y = df["diabetes"]
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        X = scaler.fit_transform(X)
        return X, y
    else:
        raise FileNotFoundError("File not found. Please upload the dataset.")

def train_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    return reg_model, X_test, y_test