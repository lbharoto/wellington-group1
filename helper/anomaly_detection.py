"""
File: anomaly_detection.py
Description: This file contains the implementation of the AnomalyDetection class, which is used to detect anomalies in a dataset.
The class provides methods to detect anomalies and plot them.

Classes:
    AnomalyDetection: A class for detecting anomalies in a dataset and plotting them.

Functions:
    None

Usage:
    from anomaly_detection import AnomalyDetection

    # Create an instance of AnomalyDetection with a pandas DataFrame containing 'prediction' and 'actual' columns
    anomaly_detector = AnomalyDetection(data)

    # Detect anomalies in the dataset
    anomaly_detector.detect_anomalies()

    # Plot actual data with detected anomalies
    anomaly_detector.plot_anomalies()
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

class AnomalyDetection:
    """
    AnomalyDetection class provides methods for detecting anomalies in a dataset and plotting them.

    Attributes:
        data (pd.DataFrame): The input dataset containing 'prediction' and 'actual' columns.

    Methods:
        detect_anomalies():
            Detects anomalies in the dataset and adds 'error', 'zscore', and 'anomaly' columns to the data.

        plot_anomalies():
            Plots the actual data with detected anomalies highlighted in red.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes an instance of the AnomalyDetection class.

        Args:
            data (pd.DataFrame): The input dataset containing 'prediction' and 'actual' columns.
        """
        self.data = data

    def detect_anomalies(self):
        """
        Detects anomalies in the dataset and adds 'error', 'zscore', and 'anomaly' columns to the data.
        Anomalies are detected based on z-scores of prediction errors.
        """
        self.data.replace([np.inf, -np.inf], np.NaN, inplace=True)
        self.data.fillna(0, inplace=True)
        self.data['error'] = self.data['prediction'] - self.data['actual']
        self.data['zscore'] = zscore(self.data['error'])
        self.data['anomaly'] = np.where((self.data['zscore'] > 3) | (self.data['zscore'] < -3), True, False)

    def plot_anomalies(self):
        """
        Plots the actual data with detected anomalies highlighted in red.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['actual'], color='black', label='Actuals')
        plt.scatter(self.data.index[self.data['anomaly']], self.data['actual'][self.data['anomaly']],
                    color='red', label='Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Actuals and Anomalies')
        plt.legend()
        plt.show()
