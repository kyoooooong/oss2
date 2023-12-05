import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

def sort_dataset(dataset_df):
    arranged = dataset_df.sort_values('year')
    return arranged

def split_dataset(dataset_df):
    dataset_df['salary'] = dataset_df['salary'] * 0.001

    X_data = dataset_df.drop('salary', axis=1)
    Y_data = dataset_df['salary']

    X_train = X_data.iloc[:1718]
    Y_train = Y_data.iloc[:1718]
    X_test = X_data.iloc[1718:]
    Y_test = Y_data.iloc[1718:]

    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
    numerical_columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']

    numerical_df = dataset_df[numerical_columns]

    return numerical_df


def train_predict_decision_tree(X_train, Y_train, X_test):
    tree_regressor = DecisionTreeRegressor()
    tree_regressor.fit(X_train, Y_train)
    tree_prediction = tree_regressor.predict(X_test)
    return tree_prediction

def train_predict_random_forest(X_train, Y_train, X_test):
    forest_regressor = RandomForestRegressor()
    forest_regressor.fit(X_train, Y_train)
    forest_prediction = forest_regressor.predict(X_test)
    return forest_prediction

def train_predict_svm(X_train, Y_train, X_test):
    svm_pipeline = make_pipeline(StandardScaler(), SVR())
    svm_pipeline.fit(X_train, Y_train)
    svm_prediction = svm_pipeline.predict(X_test)
    return svm_prediction

def calculate_RMSE(labels, predictions):
    mse = mean_squared_error(labels, predictions)
    rmse = sqrt(mse)
    return rmse


# TODO: Implement this function

if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))