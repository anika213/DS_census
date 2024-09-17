import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# Function to load data
def load_in_data():
    census_income = fetch_ucirepo(id=20)
    X = census_income.data.features
    y = census_income.data.targets
    data = pd.concat([X, y], axis=1)
    # Fix any discrepancies in the target column's values
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    return data

# Function to process the data and build a decision tree with cross-validation
def decision_tree_with_cross_validation(df):
    label_encoders = {}
    # Apply label encoding to all categorical columns
    categorical_columns = ['workclass', 'occupation', 'education', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'income']

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])  # Convert each categorical column to numeric values
        label_encoders[column] = le  # Store the encoder if needed later for inverse transforms

    # Separate features (X) and target (y)
    X = df.drop('income', axis=1)  # Features: everything except 'income'
    y = df['income']  # Target: 'income'

    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Perform K-Fold Cross Validation (Evaluate using cross-validation)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold Cross-Validation
    cv_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')

    # Output Cross-Validation results
    print(f'Cross-Validation Accuracy Scores: {cv_scores}')
    print(f'Mean CV Accuracy: {np.mean(cv_scores) * 100:.2f}%')
    print(f'Standard Deviation of CV Accuracy: {np.std(cv_scores) * 100:.2f}%')

    # Now, split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model to the training data
    clf.fit(X_train, y_train)

    # Accuracy on the training set
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

    # Accuracy on the test set
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Main function to load data and run decision tree with cross-validation
def main():
    data = load_in_data()  # Load the dataset
    print(data.info())  # Print info to see the structure of the data
    decision_tree_with_cross_validation(data)  # Build and evaluate the decision tree with cross-validation

# Run the main function if this script is executed
if __name__ == '__main__':
    main()
