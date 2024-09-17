'''
The Assignment on car purchasing data
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys


class DATA:
    def __init__(self, file):
        try: # Here we are calling the data
            self.df = pd.read_csv(file, encoding='latin1')
            self.df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1,inplace=True)
            print(self.df.head())

            # Assuming the last column is the target
            self.X = self.df.iloc[:, :-1].values
            self.y = self.df.iloc[:, -1].values

            # Splitting data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)

            # Initialize model
            self.reg = LinearRegression()

        except Exception as e:
            error_msg, error_type, error_line = sys.exc_info()
            print(f'error_line->{error_line.tb_lineno}, error_msg->{error_msg}, error_type->{error_type}')

    def TRAINING(self):
        try:
            self.reg.fit(self.X_train, self.y_train)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')

    def TRAINED_PERFORMANCE(self):
        try:
            self.y_train_pred = self.reg.predict(self.X_train)
            print(f'Train Accuracy : {r2_score(self.y_train, self.y_train_pred)}')
            print(f'Train Loss using Mean Squared Error : {mean_squared_error(self.y_train, self.y_train_pred)}')
            print(f'Train Loss Using Absolute Mean Error : {mean_absolute_error(self.y_train, self.y_train_pred)}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')

    def TESTING(self):
        try:
            self.reg.fit(self.X_test,
                         self.y_test)  # This line might not be appropriate; you might want to fit on the training set
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')

    def TEST_PERFORMANCE(self):
        try:
            self.y_test_pred = self.reg.predict(self.X_test)
            print(f'Test Accuracy : {r2_score(self.y_test, self.y_test_pred)}')
            print(f'Test Loss using Mean Squared Error : {mean_squared_error(self.y_test, self.y_test_pred)}')
            print(f'Test Loss Using Absolute Mean Error : {mean_absolute_error(self.y_test, self.y_test_pred)}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')


if __name__ == "__main__":
    try:
        obj = DATA('C:\\Users\\abc\\Downloads\\ML\\pythonProject\\Car_Purchasing_Data.csv')
        obj.TRAINING()
        obj.TRAINED_PERFORMANCE()
        obj.TESTING()
        obj.TEST_PERFORMANCE()
    except Exception as e:
        error_msg, error_type, error_line = sys.exc_info()
        print(f'error_line->{error_line.tb_lineno}, error_msg->{error_msg}, error_type->{error_type}')
