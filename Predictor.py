"""
Description: This is a Loan Default Probability predictor based on a Random Forest Model

Packages requirements: Scientific python stack (numpy, pandas, scikit-learn) + imblearn

Usage: Two data inputs are used: puzzle_train_dataset.csv and puzzle_test_dataset.csv
       An output file named predictions in .csv format is generated
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Full File Path may be required
train_data = 'puzzle_train_dataset.csv'
test_data = 'puzzle_test_dataset.csv'

def train_import(train_data):
    df = pd.read_csv(train_data, encoding='utf-8')
    return df

def test_import(test_data):
    df = pd.read_csv(test_data, encoding='utf-8')
    return df

def feature_manip(df, features):
    # Impute Missing Values Based on the Mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', copy=False)
    df['income'] = imputer.fit_transform(df['income'].values.reshape(-1, 1))
    df['risk_rate'] = imputer.fit_transform(df['risk_rate'].values.reshape(-1, 1))
    df['n_accounts'] = imputer.fit_transform(df['n_accounts'].values.reshape(-1, 1))
    df['amount_borrowed'] = imputer.fit_transform(df['amount_borrowed'].values.reshape(-1, 1))

    # Impute Missing Values Based on the Most Frequent Value
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent', copy=False)
    df['credit_limit'] = imputer.fit_transform(df['credit_limit'].values.reshape(-1,1))
    df['borrowed_in_months'] = imputer.fit_transform(df['borrowed_in_months'].values.reshape(-1, 1))

    # Create New Variables 'amount_v_income', 'risk', and 'risk_amount_income'
    df['amount_v_income'] = df.apply(lambda row: (row.amount_borrowed / row.income), axis=1)
    df['risk'] = df.apply(lambda row: (row.risk_rate * row.borrowed_in_months), axis = 1)
    df['risk_amount_income'] = df.apply(lambda row: (row.risk * row.amount_v_income), axis=1)

    # Creates a boolean variable for Facebook profiles and fills the missing values with 0 (no profile)
    df['facebook_profile'] = df['facebook_profile'].astype('bool').fillna(0)

    # Creates Dummy Variables for several Categorical Variables
    df = pd.get_dummies(df, columns=['sign','gender', 'state', 'real_state'])

    return df.loc[:, features]

def split_xy(X):
    return X.drop('default', axis=1), X['default'].astype('bool')

def test_predict_proba(df, features, rf):
    X = feature_manip(df, ['ids'] + features)
    X['predictions'] = rf.predict_proba(X.iloc[:,1:].values)[:,1]
    return X.loc[:,['ids','predictions']]

# Import the Puzzle Train data set
df = train_import(train_data)

# Remove Samples Where Target Variable ('default') is missing
df.dropna(axis=0, how='any', subset=['default'], inplace=True)

# Select Features that are Interpretable and Have Minimal Missing Values
features = ['default', 'risk_amount_income', 'amount_borrowed', 'income', 'credit_limit', 'facebook_profile',  'gender_m', 'gender_f', 'n_accounts', 'real_state_N5/CE7lSkAfB04hVFFwllw==', 'real_state_n+xK9CfX0bCn77lClTWviw==', 'sign_sagi', 'state_xsd3ZdsI3356I3xMxZeiqQ==', 'sign_taur']

# Apply Feature Manipulation to the DataFrame
X = feature_manip(df, features)

# Remove 'Default' from the Features
features = features[1:]

# Split the DF into Features (X) and Label (y)
X, y = split_xy(X)

# Instantiate SMOTE and Split the df into Training and Test Sets
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
columns = X_train.columns

# Fit the X,y training sets to oversampled versions
os_data_X, os_data_y = os.fit_sample(X_train, np.ravel(y_train))
os_data_X = pd.DataFrame(data = os_data_X, columns=columns)
os_data_y = pd.DataFrame(data = os_data_y, columns =['default'])

# Check the Numbers of the df
if len(os_data_y[os_data_y['default']==1])/len(os_data_X) != len(os_data_y[os_data_y['default']==0])/len(os_data_X):
    print("Data set is imbalanced. Consider adjusting.\n")

# Instantiate the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=800, max_depth=100, bootstrap=True, random_state=0)

# Split the oversampled dfs into training and test sets
X_train, X_test, y_train, y_test = train_test_split(os_data_X,os_data_y, test_size=0.2, random_state=0)
# Fit the Random Forest Classifier model
rf.fit(X_train, np.ravel(y_train))

# Import the Puzzle_test data set
df = test_import(test_data)

# Apply Feature Manipulations and Predictions on the Puzzle Test Set
predictions = test_predict_proba(df, features, rf)

# Print to CSV
pd.DataFrame(predictions).to_csv('predictions.csv', encoding='utf-8', index=False)
