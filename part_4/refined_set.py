import matplotlib.pyplot as plt
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
from imblearn.over_sampling import SMOTE


def load_in_data():
    census_income = fetch_ucirepo(id=20)
    X = census_income.data.features
    y = census_income.data.targets
    data = pd.concat([X, y], axis=1)
    
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    
    return data


def preprocess_data(data):
    data = data.dropna()
    
    X = data.drop(['income', 'sex', 'race'], axis=1)
    y = data['income']
    
    y = y.map({'<=50K': 0, '>50K': 1})
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols)

    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y


def build_model(input_dim):
    model = Sequential()
    model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def train_model(data):
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    model = build_model(input_dim=X_train.shape[1])
    
    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    classification_reports = {}
    
    for race in data['race'].unique():
        subgroup_condition = (data['race'] == race) 
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
    metrics = ['precision', 'recall', 'f1-score']
    num_races = len(classification_reports)
    
    fig, axes = plt.subplots(1, num_races + 1, figsize=(15, 6), sharey=True)
    
    for i, (race, report) in enumerate(classification_reports.items()):
        ax = axes[i]
        categories = ['0', '1']
        data = {metric: [report[category][metric] for category in categories] for metric in metrics}
        df = pd.DataFrame(data, index=categories)
        df.plot(kind='bar', ax=ax, title=f'Race: {race}', rot=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.legend(loc='upper right')
        ax.set_xticklabels(categories, rotation=45, ha='right')
    
    ax = axes[-1]
    races = list(classification_reports.keys())
    accuracy_values = [classification_reports[race]['accuracy'] for race in races]
    ax.bar(races, accuracy_values, color='b')
    ax.set_title('Accuracy')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy Score')
    ax.set_xticklabels(races, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()



def plot_loss_accuracy(history):
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
    data = load_in_data()
    history = train_model(data)
    plot_loss_accuracy(history)


if __name__ == '__main__':
    main()
