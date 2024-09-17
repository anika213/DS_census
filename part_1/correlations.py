import seaborn as sns
from scipy.stats import pointbiserialr
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np
import os


def load_in_data():
    census_income = fetch_ucirepo(id=20)    
    X = census_income.data.features 
    y = census_income.data.targets 
    data = pd.concat([X, y], axis=1)
    data = data.replace(' ?', pd.NA)
    # Drop rows with missing values
    data = data.dropna()
    # disp;lay number of rows
    print(data.head())
    

    return data

def point_biserial_analysis(data, binary_column, continuous_columns):

    data[binary_column] = data[binary_column].apply(lambda x: 0 if x in ['<=50K', '<=50K.'] else 1)
    correlations = {}

    for col in continuous_columns:
        correlation, p_value = pointbiserialr(data[binary_column], data[col])
        correlations[col] = (correlation, p_value)

    correlation_df = pd.DataFrame(correlations, index=['Correlation', 'P-value']).T

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_df[['Correlation']], annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f"Point-Biserial Correlation Heatmap ({binary_column})")
    plt.show()
    return correlation_df


def correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('feature_analysis/correlation_matrix.png')


def calculate_covariance(data):
    # first we need to encode income so that we don't have categorical data
    data['income_num'] = data['income'].map({' <=50K': 0, ' >50K': 1})
    print(f"Data type of 'age': {data['age'].dtype}")
    print(f"Data type of 'income_num': {data['income_num'].dtype}")
    print(data['income_num'].head())
    covariance_matrix = np.cov(data['income_num'], data['age'])
    covariance = covariance_matrix[0, 1]
    print("Covariance matrix:")
    print(covariance_matrix)
    print(f"\nCovariance between income and age: {covariance}")


def main():
    data = load_in_data()
    correlation_matrix(data)
    # calculate_covariance(data)

    # calculate correlation 
    continuous_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']  # Add other continuous columns as necessary
    result = point_biserial_analysis(data, binary_column='income', continuous_columns=continuous_columns)
    print(result)



if __name__ == '__main__':
    main()
