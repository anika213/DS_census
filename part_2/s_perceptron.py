import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np
import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



def load_in_data():
    # Fetching the dataset
    census_income = fetch_ucirepo(id=20)
    X = census_income.data.features
    y = census_income.data.targets
    data = pd.concat([X, y], axis=1)
    
    # Replace specific income categories with generalized form
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})

    return data


def preprocess_data(data):
    # Handle missing values by dropping rows with NaNs
    data = data.dropna()
    
    # Separate features and target variable
    X = data.drop('income', axis=1)
    y = data['income']
    
    # Encode the target variable
    y = y.map({'<=50K': 0, '>50K': 1})
    
    # Identify categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y


def build_model(input_dim):
    # Build the model
    model = Sequential()
    model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def train_model(data):
    X, y = preprocess_data(data)
    print(data['race'].head())
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    
    # Build the model
    model = build_model(input_dim=X_train.shape[1])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Reduced epochs for brevity
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    # visualise the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")

    print(accuracy_score(y_test, y_pred))
    return history


def plot_loss_accuracy(history):
    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess the data
    data = load_in_data()
    # print(data.info())
    # print(data.head())
    history = train_model(data)
    plot_loss_accuracy(history)
    train_model(data)

   


if __name__ == '__main__':
    main()
