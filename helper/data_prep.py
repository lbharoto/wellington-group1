"""
File: credit_data_processor.py
Description: This file contains the implementation of the CreditDataProcessor class,
             which is used to process credit data by loading, manipulating, and analyzing it.

Classes:
    CreditDataProcessor: A class for loading, processing, and analyzing credit data.

Usage:
    from credit_data_processor import CreditDataProcessor

    # Create an instance of CreditDataProcessor with a file path
    credit_processor = CreditDataProcessor(file_path='credit_data.csv', train_ratio=0.8)

    # Load data from the CSV file
    credit_processor.load_data()

    # Get data for a specific company
    credit_processor.get_company_data(company_code='company_A')

    # Add seasonality information to the data
    seasonality_data = credit_processor.add_seasonality()
"""

import numpy as np
import pandas as pd

class CreditDataProcessor:
    """
    CreditDataProcessor class provides methods for loading, processing, and analyzing credit data.

    Attributes:
        file_path (str): The path to the CSV file containing credit data.
        train_ratio (float): The ratio of data to be used for training.

    Methods:
        load_data():
            Loads credit data from the CSV file and prepares it for processing.

        get_company_data(company_code: str):
            Retrieves data for a specific company and divides it into training and testing sets.

        add_seasonality() -> pd.DataFrame:
            Adds seasonality information to the company's data and returns the updated DataFrame.
    """

    def __init__(self, file_path: str, train_ratio: float = 0.8):
        """
        Initializes an instance of the CreditDataProcessor class.

        Args:
            file_path (str): The path to the CSV file containing credit data.
            train_ratio (float): The ratio of data to be used for training.
        """
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.credit = None
        self.pivot_credit = None
        self.data = None
        self.train = None
        self.test = None
    
    def load_data(self):
        """
        Loads credit data from the CSV file and prepares it for processing.
        """
        self.credit = pd.read_csv(self.file_path)
        self.credit.columns = ['company', 'date', 'data']
        self.credit.date = pd.to_datetime(self.credit.date)
        self.pivot_credit = pd.pivot_table(self.credit, values="data", index="date", columns="company")
        self.pivot_credit = self.pivot_credit.fillna(0)
    
    def get_company_data(self, company_code: str):
        """
        Retrieves data for a specific company and divides it into training and testing sets.

        Args:
            company_code (str): The code representing the company.

        Returns:
            None
        """
        if self.pivot_credit is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        data = self.pivot_credit[company_code]
        data = data.reset_index()
        data = data.rename(columns={"date": "ds", company_code: "y"})
        self.data = data
        
        cutoff = int(np.round(len(data) * self.train_ratio))
        self.train = data[:cutoff]
        self.test = data[cutoff:]

    def add_seasonality(self) -> pd.DataFrame:
        """
        Adds seasonality information to the company's data and returns the updated DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with added seasonality information.
        """
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        data = self.data.copy()

        data['year'] = data['ds'].dt.year
        data['month'] = data['ds'].dt.month
        data['day'] = data['ds'].dt.day

        data = data.drop(columns=["ds"])
        data = data.set_index(["month", "day"])

        pivot_df = data.pivot_table(index=['month', 'day'], columns='year', values='y')

        def find_mean(row):
            values = row.dropna()  # Remove NaN values before calculating Euclidean distance
            return np.mean(values)

        pivot_df['seasonality'] = pivot_df.apply(find_mean, axis=1)

        # Unpivot (melt) the DataFrame
        pivot_df = pivot_df.reset_index()
        melted_df = pd.melt(pivot_df, id_vars=['month', 'day'], var_name='year', value_name='value')

        # Separate the DataFrame into two DataFrames: one for 'value' and one for 'seasonality'
        value_df = melted_df[melted_df['year'] != 'seasonality']
        distance_df = melted_df[melted_df['year'] == 'seasonality']

        # drop Feb 29
        condition1 = (value_df['day'] == 29) & (value_df['month'] == 2) & (value_df['year'] == 2017)
        condition2 = (value_df['day'] == 29) & (value_df['month'] == 2) & (value_df['year'] == 2018)
        condition3 = (value_df['day'] == 29) & (value_df['month'] == 2) & (value_df['year'] == 2019)
        condition4 = (value_df['day'] == 29) & (value_df['month'] == 2) & (value_df['year'] == 2021)
        condition5 = (value_df['day'] == 29) & (value_df['month'] == 2) & (value_df['year'] == 2022)

        # Drop rows based on the condition
        value_df = value_df.drop(index=value_df[condition1].index)
        value_df = value_df.drop(index=value_df[condition2].index)
        value_df = value_df.drop(index=value_df[condition3].index)
        value_df = value_df.drop(index=value_df[condition4].index)
        value_df = value_df.drop(index=value_df[condition5].index)

        # Create a new column 'date' with the format "YYYY-MM-DD"
        value_df['date'] = pd.to_datetime(value_df[['year', 'month', 'day']])
        value_df = value_df.drop(columns=['month', 'day', 'year'])


        # Create a list of years from 2017 to 2022
        years = list(range(2016, 2023))

        # Create an empty DataFrame to store the expanded data
        distance_expanded_df = pd.DataFrame()

        # Iterate through each row in the original DataFrame
        for index, row in distance_df.iterrows():
            # Repeat the current row for each year and append it to the expanded DataFrame
            for year in years:
                new_row = row.copy()  # Create a copy of the current row
                new_row['year'] = int(year)  # Set the 'year' column to the current year
                distance_expanded_df = distance_expanded_df.append(new_row, ignore_index=True)

        # drop Feb 29
        condition1 = (distance_expanded_df['day'] == 29) & (distance_expanded_df['month'] == 2) & (distance_expanded_df['year'] == 2017)
        condition2 = (distance_expanded_df['day'] == 29) & (distance_expanded_df['month'] == 2) & (distance_expanded_df['year'] == 2018)
        condition3 = (distance_expanded_df['day'] == 29) & (distance_expanded_df['month'] == 2) & (distance_expanded_df['year'] == 2019)
        condition4 = (distance_expanded_df['day'] == 29) & (distance_expanded_df['month'] == 2) & (distance_expanded_df['year'] == 2021)
        condition5 = (distance_expanded_df['day'] == 29) & (distance_expanded_df['month'] == 2) & (distance_expanded_df['year'] == 2022)

        # Drop rows based on the condition
        distance_expanded_df = distance_expanded_df.drop(index=distance_expanded_df[condition1].index)
        distance_expanded_df = distance_expanded_df.drop(index=distance_expanded_df[condition2].index)
        distance_expanded_df = distance_expanded_df.drop(index=distance_expanded_df[condition3].index)
        distance_expanded_df = distance_expanded_df.drop(index=distance_expanded_df[condition4].index)
        distance_expanded_df = distance_expanded_df.drop(index=distance_expanded_df[condition5].index)

        # Construct date columns from year, month and day columns
        distance_expanded_df['date'] = pd.to_datetime(distance_expanded_df[['year', 'month', 'day']])
        distance_expanded_df = distance_expanded_df.drop(columns=['month', 'day', 'year'])

        # Sort dataframe based on date column
        distance_expanded_df = distance_expanded_df.sort_values(by='date', ascending=True)
        distance_expanded_df = distance_expanded_df.reset_index(drop=True)

        distance_expanded_df = distance_expanded_df.rename(columns={"date":"ds", "value":"seasonality"})
        distance_expanded_df = distance_expanded_df.set_index('ds')

        value_df = value_df.rename(columns={"date":"ds", "value":"actual"})
        value_df = value_df.set_index('ds')

        merged_df = distance_expanded_df.merge(value_df, left_index=True, right_index=True)
        merged_df = merged_df.iloc[:len(data), :]

        return merged_df
