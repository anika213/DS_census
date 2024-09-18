import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_data():
    data = fetch_ucirepo(id=20)
    X = data.data.features
    y = data.data.targets
    data = pd.concat([X, y], axis=1)
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    return data

def calculate_percentages(data):
    # Split data to only look at those with a bachelor's degree
    data = data[data['sex'] == 'Female']
    data = data[data['race'] != 'White']
    print(data['income'].value_counts(normalize=True) * 100)
    print(data.head())
    print(len(data))
    
    count = 0
    # Loop through the dataframe to find percentage of people with income greater than 50k
    for _, row in data.iterrows():
        if row['income'] == ">50K":
            count += 1
            
    print(f"Percentage of people with income greater than 50k: {count/len(data) * 100:.2f}%")


def main():
    data = load_data()
    print(data.head())
    calculate_percentages(data)



if __name__ == '__main__':
    main()
