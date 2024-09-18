import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ucimlrepo import fetch_ucirepo

# Function to load data
def load_in_data():
    census_income = fetch_ucirepo(id=20)
    X = census_income.data.features
    y = census_income.data.targets
    data = pd.concat([X, y], axis=1)
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    return data


def decision_tree_with_cross_validation(df):
    label_encoders = {}
    # label encode the categorical columns
    categorical_columns = ['workclass', 'occupation', 'education', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'income']

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column]) 
        label_encoders[column] = le  

   
    X = df.drop('income', axis=1)  # Features: everything except 'income'
    y = df['income']  # Target: 'income'

   
    clf = DecisionTreeClassifier(random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)  
    cv_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')


    print(f'Cross-Validation Accuracy Scores: {cv_scores}')
    print(f'Mean CV Accuracy: {np.mean(cv_scores) * 100:.2f}%')
    print(f'Standard Deviation of CV Accuracy: {np.std(cv_scores) * 100:.2f}%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    clf.fit(X_train, y_train)


    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

# accuracy of test data
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    data = load_in_data()  #
    print(data.info())  
    decision_tree_with_cross_validation(data)  


if __name__ == '__main__':
    main()
