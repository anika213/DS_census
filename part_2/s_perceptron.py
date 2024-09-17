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
    # Handling missing values by filling with the most frequent value
    data.fillna(data.mode().iloc[0], inplace=True)

    # Convert the target column 'income' to binary (0 for <=50K, 1 for >50K)
    data['income'] = data['income'].apply(lambda x: 1 if '>50K' in x else 0)

    # Label Encoding for categorical variables
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Selecting features and target
    X = data.drop('income', axis=1)
    y = data['income']

    # Normalize/standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def build_single_perceptron_model(input_dim):
    # Initialize the perceptron model
    model = Sequential()

    # Add a single neuron with a sigmoid activation function
    model.add(Dense(1, activation='sigmoid', input_dim=input_dim))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def main():
    # Load and preprocess the data
    data = load_in_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Build the model
    model = build_single_perceptron_model(X_train.shape[1])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy}')

    # Make predictions and print the classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
