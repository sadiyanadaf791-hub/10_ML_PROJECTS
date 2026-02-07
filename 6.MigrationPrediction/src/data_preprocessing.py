import pandas as pd

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)

    # Encode Measure column
    data['Measure'] = data['Measure'].map({
        'Arrivals': 0,
        'Departures': 1,
        'Net': 2
    })

    # Factorize categorical columns
    data['CountryID'] = pd.factorize(data['Country'])[0]
    data['CitID'] = pd.factorize(data['Citizenship'])[0]

    # Handle missing values
    data['Value'].fillna(data['Value'].median(), inplace=True)

    # Drop original categorical columns
    data.drop(['Country', 'Citizenship'], axis=1, inplace=True)

    X = data[['CountryID', 'Measure', 'Year', 'CitID']]
    y = data['Value']

    return X, y
