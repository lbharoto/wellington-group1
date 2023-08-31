"""
File: train_test.py
Description: This file contains the implementation of three classes for different time series forecasting models.

Classes:
    DataSplitter: A class for splitting time series data into training and testing sets.
    NewModel: A class for building and using a new forecasting model based on Linear Regression.
    SARIMAXModel: A class for building and using a SARIMA-X model for time series forecasting.
    ProphetModel: A class for building and using a Prophet model for time series forecasting.
"""
import cmdstanpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm


class DataSplitter:
    """
    DataSplitter class provides methods for splitting time series data into training and testing sets.

    Attributes:
        dataframe (pd.DataFrame): The input time series data.
        train_ratio (float): The ratio of data to be used for training.

    Methods:
        split_data() -> (pd.DataFrame, pd.DataFrame):
            Splits the data into training and testing sets and returns both.
    """
    def __init__(self, dataframe, train_ratio):
        """
        Initializes an instance of the DataSplitter class.

        Args:
            dataframe (pd.DataFrame): The input time series data.
            train_ratio (float): The ratio of data to be used for training.
        """
        self.dataframe = dataframe
        self.train_ratio = train_ratio
        self.train_samples = int(len(dataframe) * train_ratio)
        self.test_samples = len(dataframe) - self.train_samples

    def split_data(self):
        """
        Splits the data into training and testing sets and returns both.

        Returns:
            pd.DataFrame: The training data.
            pd.DataFrame: The testing data.
        """
        train_data = self.dataframe[:self.train_samples].reset_index()
        test_data = self.dataframe[-self.test_samples:].reset_index()
        return train_data, test_data
    
class NewModel:
    """
    NewModel class provides methods for building and using a new forecasting model based on Linear Regression.

    Attributes:
        data (pd.DataFrame): The input time series data.
        period (int): The period for STL decomposition.
        seasonal (int): The seasonal parameter for STL decomposition.
        independent_cols (list): List of column names for independent variables.
        dependent_col (str): The column name for the dependent variable.

    Methods:
        get_trend():
            Computes and adds the trend component to the data.

        train_linear_regression_model():
            Trains a Linear Regression model on the trend and independent variables.

        predict(test_data: pd.DataFrame) -> pd.DataFrame:
            Makes predictions using the trained model and returns the predictions.

        calculate_mean_squared_error(actual, prediction) -> float:
            Calculates the Mean Squared Error (MSE) between actual and predicted values.

        plot_results(test_data: pd.DataFrame):
            Plots the actual and predicted values.
    """
    def __init__(self, data, period, seasonal, independent_cols, dependent_col):
        """
        Initializes an instance of the NewModel class.

        Args:
            data (pd.DataFrame): The input time series data.
            period (int): The period for STL decomposition.
            seasonal (int): The seasonal parameter for STL decomposition.
            independent_cols (list): List of column names for independent variables.
            dependent_col (str): The column name for the dependent variable.
        """
        self.data = data
        self.period = period
        self.seasonal = seasonal
        self.independent_cols = independent_cols
        self.dependent_col = dependent_col
        self.model = None  # Store the trained Linear Regression model

    def get_trend(self):
        """
        Computes and adds the trend component to the data.
        """
        stl = STL(self.data["actual"], period=self.period, seasonal=self.seasonal)
        result = stl.fit()
        trend = result.trend
        self.data["trend"] = trend

    def train_linear_regression_model(self):
        """
        Trains a Linear Regression model on the trend and independent variables.
        """
        self.data['days'] = (self.data['ds'] - self.data['ds'].min()).dt.days + 1
        X = self.data[['days'] + self.independent_cols]
        y = self.data['trend']
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict(self, test_data):
        """
        Makes predictions using the trained model and returns the predictions.

        Args:
            test_data (pd.DataFrame): The testing data.

        Returns:
            pd.DataFrame: The testing data with added prediction columns.
        """
        test_data["days"] = len(self.data) + (test_data['ds'] - test_data['ds'].min()).dt.days + 1
        test_data["prediction_trend"] = self.model.predict(test_data[["days", "holiday", "events"]])
        test_data["prediction"] = test_data["prediction_trend"] + test_data["seasonality"] - np.mean(test_data["seasonality"])
        return test_data

    def calculate_mean_squared_error(self, actual, prediction):
        """
        Calculates the Mean Squared Error (MSE) between actual and predicted values.

        Args:
            actual (pd.Series): The actual values.
            prediction (pd.Series): The predicted values.

        Returns:
            float: The Mean Squared Error (MSE).
        """
        return mean_squared_error(actual, prediction)
    
    def plot_results(self, test_data):
        """
        Plots the actual and predicted values.

        Args:
            test_data (pd.DataFrame): The testing data.
        """
        figsize = (12, 7)
        plt.figure(figsize=figsize)
        plt.plot(test_data.set_index("ds")["actual"], label='actual data')
        plt.plot(test_data.set_index("ds")["prediction"], label='prediction')
        plt.legend(loc='upper right')
        plt.show()

class SARIMAXModel:
    """
    SARIMAXModel class provides methods for building and using a SARIMA-X model for time series forecasting.

    Attributes:
        train (pd.Series): The training data.
        test (pd.Series): The testing data.
        my_order (tuple): The (p, d, q) order of the SARIMA model.
        my_seasonal_order (tuple): The (P, D, Q, S) seasonal order of the SARIMA model.

    Methods:
        fit_predict():
            Fits the SARIMAX model to training data and makes predictions for testing data.

        evaluate_errors():
            Evaluates the model's performance using Mean Squared Error (MSE).

        plot_results():
            Plots the actual and predicted values.
    """
    def __init__(self, train, test, my_order, my_seasonal_order):
        """
        Initializes an instance of the SARIMAXModel class.

        Args:
            train (pd.Series): The training data.
            test (pd.Series): The testing data.
            my_order (tuple): The (p, d, q) order of the SARIMA model.
            my_seasonal_order (tuple): The (P, D, Q, S) seasonal order of the SARIMA model.
        """
        self.train = train
        self.test = test
        self.my_order = my_order
        self.my_seasonal_order = my_seasonal_order
        self.history = [x for x in self.train]
        self.predictions = []

    def fit_predict(self):
        """
        Fits the SARIMAX model to training data and makes predictions for testing data.
        """
        for t in range(len(self.test)):
            model = sm.tsa.SARIMAX(
                self.history,
                order=self.my_order,
                seasonal_order=self.my_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            self.predictions.append(output[0])

            self.history.append(output[0])

    def evaluate_errors(self):
        """
        Evaluates the model's performance using Mean Squared Error (MSE).
        """
        real_error = mean_squared_error(self.test, self.predictions)
        print('Test rmse on normal scale: %.3f' % real_error)

    def plot_results(self):
        """
        Plots the actual and predicted values.
        """
        figsize = (12, 7)
        plt.figure(figsize=figsize)
        plt.plot(self.test, label='Actuals')
        plt.plot(self.predictions, color='red', label='Predicted')
        plt.legend(loc='upper right')
        plt.show()

class ProphetModel:
    """
    ProphetModel class provides methods for building and using a Prophet model for time series forecasting.

    Attributes:
        holidays_df (pd.DataFrame): DataFrame containing holiday dates and names.
        train_ratio (float): The ratio of data to be used for training.

    Methods:
        fit(train_data: pd.DataFrame):
            Fits the Prophet model to training data.

        predict(train_data: pd.DataFrame, test_data: pd.DataFrame):
            Makes predictions using the Prophet model and evaluates the results.

        calculate_mse() -> float:
            Calculates the Mean Squared Error (MSE) for the Prophet model predictions.

        plot_results():
            Plots the actual and predicted values.
    """
    def __init__(self, holidays_df, train_ratio):
        """
        Initializes an instance of the ProphetModel class.

        Args:
            holidays_df (pd.DataFrame): DataFrame containing holiday dates and names.
            train_ratio (float): The ratio of data to be used for training.
        """
        self.model = Prophet(holidays=holidays_df)
        self.forecast = None
        self.prediction = None
        self.actual = None
        self.eval_df = None
        self.train_ratio = train_ratio

    def fit(self, train_data):
        """
        Fits the Prophet model to training data.

        Args:
            train_data (pd.DataFrame): The training data.
        """
        self.forecast = self.model.fit(train_data)
    
    def predict(self, train_data, test_data):
        """
        Makes predictions using the Prophet model and evaluates the results.

        Args:
            train_data (pd.DataFrame): The training data.
            test_data (pd.DataFrame): The testing data.
        """
        future = self.model.make_future_dataframe(periods=len(test_data))
        self.forecast = self.model.predict(future)
        self.forecast = self.forecast.set_index("ds")
        
        prediction = self.forecast[["yhat"]][int(np.round(len(train_data) * self.train_ratio)):]
        self.prediction = prediction
        self.actual = test_data.set_index("ds")
        
        self.eval_df = self.actual.merge(self.prediction, left_index=True, right_index=True)
        self.eval_df = self.eval_df.rename(columns={"y": "actual", "yhat": "prediction"})
        
    def calculate_mse(self):
        """
        Makes predictions using the Prophet model and evaluates the results.

        Args:
            train_data (pd.DataFrame): The training data.
            test_data (pd.DataFrame): The testing data.
        """
        mse = mean_squared_error(self.eval_df["actual"], self.eval_df["prediction"])
        return mse
    
    def plot_results(self):
        """
        Plots the actual and predicted values.
        """
        figsize = (12, 7)
        plt.figure(figsize=figsize)
        plt.plot(self.eval_df["actual"], label='actual data')
        plt.plot(self.eval_df["prediction"], label='prediction')
        plt.legend(loc='upper right')
        plt.show()