from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_in_data():
    census_income = fetch_ucirepo(id=20)    
    X = census_income.data.features 
    y = census_income.data.targets 
    data = pd.concat([X, y], axis=1)
    return data

def age_vs_income(data):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='income', y='age', data=data)
    plt.title('Age vs Income')
    plt.xlabel('Income')
    plt.ylabel('Age')
    plt.savefig('bi-plots/age_vs_income.png')


def education_vs_income(data):
    plt.figure(figsize=(12, 8))
    sns.countplot(x='education', hue='income', data=data)
    plt.title('Education vs Income')
    plt.xlabel('Education')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  
    plt.tight_layout()
    plt.savefig('bi-plots/education_vs_income.png')

def occupation_vs_income(data):
    plt.figure(figsize=(12, 8))
    sns.countplot(x='occupation', hue='income', data=data)
    plt.title('Occupation vs Income')
    plt.xlabel('Occupation')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  
    plt.tight_layout()
    plt.savefig('bi-plots/occupation_vs_income.png')


def gender_vs_income(data):
    plt.figure(figsize=(6,4))
    sns.countplot(x='sex', hue='income', data=data)
    plt.title('Gender vs Income')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.legend(title='Income')
    plt.savefig('bi-plots/gender_vs_income.png')


def race_vs_income(data):
    plt.figure(figsize=(8,6))
    sns.countplot(y='race', hue='income', data=data, order=data['race'].value_counts().index)
    plt.title('Race vs Income')
    plt.xlabel('Count')
    plt.ylabel('Race')
    plt.legend(title='Income')
    plt.tight_layout()
    plt.savefig('bi-plots/race_vs_income.png')


def main():
    data = load_in_data()
    data = data.dropna() # get rid of rows without full responses
    data = data.reset_index(drop=True) 
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'}) 
    age_vs_income(data)
    education_vs_income(data)
    occupation_vs_income(data)
    race_vs_income(data)
    gender_vs_income(data)
if __name__ == '__main__':
    main()
    