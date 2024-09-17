from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_in_data():
    census_income = fetch_ucirepo(id=20)    
    # data (as pandas dataframes) 
    X = census_income.data.features 
    y = census_income.data.targets 
    # metadata 
    # print(census_income.metadata) 
    # variable information 
    # print(census_income.variables) 
    data = pd.concat([X, y], axis=1)
    return data


def age_distribution(data):
    plt.figure(figsize=(10,6))
    sns.histplot(data['age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('uni-plots/age_distribution.png')


def income_distribution(data):
    plt.figure(figsize=(6,4))
    sns.countplot(x='income', data=data)
    plt.title('Income Distribution')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.savefig('uni-plots/income_distribution.png')

def education_distribution(data):
    plt.figure(figsize=(10,6))
    sns.countplot(x='education', data=data)
    plt.title('Education Distribution')
    plt.xlabel('Education')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  
    plt.tight_layout()
    # save it in the uni-plots folder
    plt.savefig('uni-plots/education_distribution.png')       

def occupation_distribution(data):
    plt.figure(figsize=(12, 8))
    sns.countplot(x='occupation', data=data)
    plt.title('Occupation Distribution')
    plt.xlabel('Occupation')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  
    plt.tight_layout()       
    plt.savefig('uni-plots/occupation_distribution.png')


#load in data from UCI repo


print('Loading in data...')
data = load_in_data() 
data = data.dropna() # get rid of rows without full responses
data = data.reset_index(drop=True)
data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'}) 
print(data.info()) 


# uni-variate analysis
age_distribution(data) # plot age distribution
income_distribution(data) # plot income
education_distribution(data) # plot education
occupation_distribution(data) # plot occupation



