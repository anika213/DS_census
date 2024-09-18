import matplotlib.pyplot as plt
# Removed unused imports
import numpy as np
import seaborn as sns
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Import SMOTE for oversampling
from imblearn.over_sampling import SMOTE  # New import


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
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to balance the classes in the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Build the model
    model = build_model(input_dim=X_train.shape[1])
    
    # Train the model on the resampled data
    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=10,  # Adjust epochs as needed
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    classification_reports = {}
    
    # Loop through all the possible race subgroups
    for race in data['race'].unique():
        subgroup_condition = (data['race'] == race) & (data['income'] == ">50K")
        subgroup_indices = data.iloc[X_test.index][subgroup_condition].index
        X_test_subgroup = X_test.loc[subgroup_indices]
        y_test_subgroup = y_test.loc[subgroup_indices]
        y_pred_subgroup = (model.predict(X_test_subgroup) > 0.5).astype('int32')
        
        report = classification_report(y_test_subgroup, y_pred_subgroup, output_dict=True)
        classification_reports[race] = report
    
    plot_classification_reports(classification_reports)

    
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))
    return history



def plot_classification_reports(classification_reports):
    metrics = ['precision', 'recall', 'f1-score']  # Removed accuracy from here
    num_races = len(classification_reports)
    
    fig, axes = plt.subplots(1, num_races + 1, figsize=(15, 6), sharey=True)  # Extra subplot for accuracy
    
    # Plot precision, recall, f1-score
    for i, (race, report) in enumerate(classification_reports.items()):
        ax = axes[i]
        categories = ['0', '1']  # Class labels in binary classification
        data = {metric: [report[category][metric] for category in categories] for metric in metrics}
        df = pd.DataFrame(data, index=categories)
        df.plot(kind='bar', ax=ax, title=f'Race: {race}', rot=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.legend(loc='upper right')
        ax.set_xticklabels(categories, rotation=45, ha='right')  # Rotate x-axis labels for class labels
    
    # Plot accuracy
    ax = axes[-1]  # The last subplot for accuracy
    races = list(classification_reports.keys())
    accuracy_values = [classification_reports[race]['accuracy'] for race in races]
    ax.bar(races, accuracy_values, color='b')
    ax.set_title('Accuracy')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy Score')
    ax.set_xticklabels(races, rotation=45, ha='right')  # Rotate x-axis labels for races
    
    plt.tight_layout()
    plt.show()



def plot_loss_accuracy(history):
    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    
    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plotting Accuracy
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
    history = train_model(data)
    plot_loss_accuracy(history)


if __name__ == '__main__':
    main()
