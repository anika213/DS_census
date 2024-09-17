import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np
import os
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler



def load_in_data():
    census_income = fetch_ucirepo(id=20)    
    X = census_income.data.features 
    y = census_income.data.targets 
    data = pd.concat([X, y], axis=1)
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})

    return data

def initial_visualisation(data):
    data['income'] = data['income'].astype('category')
    sns.catplot(x='age', y='income', data=data, order=['<=50K', '>50K'])
    plt.show()
    # count the number of people with income less than 50k
    print(data['income'].value_counts())

def logistic_regression_singlevar(data):
    data['income'] = data['income'].replace({'<=50K': 0, '>50K': 1})
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=1)
    features  =[ 'age','hours-per-week','education-num','capital-gain','capital-loss']
    for i in range(len(features)):

        X_train = train_df[[features[i]]]
        y_train = train_df['income']
        
        X_test = test_df[[features[i]]]
        y_test = test_df['income']

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        # print('training data shape:', train_df.shape[0])
        # print('test data shape:', test_df.shape[0])
        # print(train_df['income'].value_counts())
        # print(test_df['income'].value_counts())
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}% {features[i]}')

        # print(classification_report(y_test, y_pred))
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(6,4))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix')
        # plt.show()

    
def logistic_regression_multivars(data):
    data['income'] = data['income'].replace({'<=50K': 0, '>50K': 1})
    
    # Selecting multiple features for logistic regression
    features = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
    
    # Train-test split
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=1)
    
    X_train = train_df[features]
    y_train = train_df['income']
    
    X_test = test_df[features]
    y_test = test_df['income']
    
    # Standardizing the features (scaling)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Classification report
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotting confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Multivar Logistic Regression)')
    plt.show()



def main():
    data = load_in_data()
    print(data.head())
    # logistic_regression(data)
    print(data.head())
    print(data.info())
    # initial_visualisation(data)
    print("SINGLEVAR")
    logistic_regression_singlevar(data)
    print("MULTIVAR")
    logistic_regression_multivars(data)
    

if __name__ == '__main__':
    main()