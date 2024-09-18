import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

def load_in_data():
    # Fetching the dataset
    census_income = fetch_ucirepo(id=20)
    X = census_income.data.features
    y = census_income.data.targets
    data = pd.concat([X, y], axis=1)
    
    # Replace specific income categories with generalized form
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})

    # Remove leading/trailing whitespace from string columns
    string_cols = data.select_dtypes(include=['object']).columns
    for col in string_cols:
        data[col] = data[col].str.strip()
    
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
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y

def build_model(input_dim):
    # Build the model
    model = Sequential()
    model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

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

def train_model(data):
    # Split data into train and test sets
    X, y = preprocess_data(data)
    X_train, X_test_full, y_train, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build the model
    model = build_model(input_dim=X_train.shape[1])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test_full, y_test_full), verbose=1)
    
    classification_reports = {}
    accuracies = []  # List to store accuracy values for each race
    



    # Loop through all the possible race subgroups
    for race in data['race'].unique():
        print(data['income'].head())
        subgroup_condition = (data['race'] == race) & (data['income'] == ">50K")

        subgroup_indices = data.iloc[X_test_full.index][subgroup_condition].index
        X_test_subgroup = X_test_full.loc[subgroup_indices]
        y_test_subgroup = y_test_full.loc[subgroup_indices]
        y_pred_subgroup = (model.predict(X_test_subgroup) > 0.5).astype('int32')
        
        # Generate classification report
        report = classification_report(y_test_subgroup, y_pred_subgroup, output_dict=True)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_subgroup, y_pred_subgroup)
        accuracies.append(accuracy)  # Add accuracy to the list
        
        # Store classification report and accuracy
        classification_reports[race] = report
        classification_reports[race]['accuracy'] = accuracy  # Add accuracy to the report
    
    # Calculate mean accuracy
    mean_accuracy = np.mean(accuracies)
    print(f"Mean Accuracy across all races: {mean_accuracy:.4f}")
    
    plot_classification_reports(classification_reports)


def main():
    # Load and preprocess the data
    data = load_in_data()

    train_model(data)

if __name__ == '__main__':
    main()
