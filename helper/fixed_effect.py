"""
File: data_processors.py
Description: This file contains the implementation of two classes, `HolidayDataProcessor` and `SalesDataProcessor`,
             which are used for processing holiday and sales data respectively.

Classes:
    HolidayDataProcessor: A class for creating a DataFrame containing holiday dates and their names.
    SalesDataProcessor: A class for creating a DataFrame containing sales events and their dates.

Usage:
    from data_processors import HolidayDataProcessor, SalesDataProcessor

    # Create an instance of HolidayDataProcessor for a specific year range
    holiday_processor = HolidayDataProcessor(start_year=2016, end_year=2023)

    # Create a DataFrame containing holiday dates and names
    holidays_df = holiday_processor.get_holidays_dataframe()

    # Create an instance of SalesDataProcessor for a specific year range
    sales_processor = SalesDataProcessor(start_year=2016, end_year=2023)

    # Create a DataFrame containing sales events and their dates
    sales_df = sales_processor.get_sales_df()
"""

import holidays
import pandas as pd


class HolidayDataProcessor:
    """
    HolidayDataProcessor class provides methods for creating a DataFrame containing holiday dates and their names.

    Attributes:
        start_year (int): The start year for generating holiday data.
        end_year (int): The end year for generating holiday data.

    Methods:
        create_dataframe():
            Creates a DataFrame containing holiday dates and their names.

        get_holidays_dataframe() -> pd.DataFrame:
            Retrieves the holiday DataFrame, creating it if not already created.
    """
    
    def __init__(self, start_year, end_year):
        """
        Initializes an instance of the HolidayDataProcessor class.

        Args:
            start_year (int): The start year for generating holiday data.
            end_year (int): The end year for generating holiday data.
        """
        self.start_year = start_year
        self.end_year = end_year
        self.us_holidays = holidays.US(years=range(start_year, end_year + 1))
        self.holiday_data = [{'ds': date, 'holiday': holiday} for date, holiday in self.us_holidays.items()]
        self.holidays_df = None
    
    def create_dataframe(self):
        """
        Creates a DataFrame containing holiday dates and their names.
        """
        self.holidays_df = pd.DataFrame(self.holiday_data)
        self.holidays_df['ds'] = pd.to_datetime(self.holidays_df['ds'])
        self.holidays_df = self.holidays_df.sort_values('ds')
        self.holidays_df = self.holidays_df.set_index("ds")
    
    def get_holidays_dataframe(self):
        """
        Retrieves the holiday DataFrame, creating it if not already created.

        Returns:
            pd.DataFrame: A DataFrame containing holiday dates and names.
        """
        if self.holidays_df is None:
            self.create_dataframe()
        return self.holidays_df
    
class SalesDataProcessor:
    """
    SalesDataProcessor class provides methods for creating a DataFrame containing sales events and their dates.

    Attributes:
        start_year (int): The start year for generating sales data.
        end_year (int): The end year for generating sales data.

    Methods:
        populate_sales_df():
            Populates the sales DataFrame with sales events and their dates.

        get_sales_df() -> pd.DataFrame:
            Retrieves the sales DataFrame.

    """
    def __init__(self, start_year, end_year):
        """
        Initializes an instance of the SalesDataProcessor class.

        Args:
            start_year (int): The start year for generating sales data.
            end_year (int): The end year for generating sales data.
        """
        self.start_date = pd.to_datetime(f"{start_year}-01-01")
        self.end_date = pd.to_datetime(f"{end_year}-12-31")
        self.dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.sales_df = pd.DataFrame(index=self.dates, columns=['events'])
        self.us_sales_days = {
            'Black Friday': ['2016-11-25', '2017-11-24', '2018-11-23', '2019-11-29', '2020-11-27', '2021-11-26', '2022-11-25', '2023-11-24'],
            'Cyber Monday': ['2016-11-28', '2017-11-27', '2018-11-26', '2019-12-02', '2020-11-30', '2021-11-29', '2022-11-28', '2023-11-27'],
            'Amazon Prime Day': ['2016-07-12', '2017-07-11', '2018-07-16', '2019-07-15', '2020-10-13', '2021-06-21', '2022-07-11', '2023-07-10'],
            'Memorial Day': ['2016-05-30', '2017-05-29', '2018-05-28', '2019-05-27', '2020-05-25', '2021-05-31', '2022-05-30', '2023-05-29'],
            'Labor Day': ['2016-09-05', '2017-09-04', '2018-09-03', '2019-09-02', '2020-09-07', '2021-09-06', '2022-09-05', '2023-09-04'],
            'President\'s Day': ['2016-02-15', '2017-02-20', '2018-02-19', '2019-02-18', '2020-02-17', '2021-02-15', '2022-02-21', '2023-02-20'],
            'Fourth of July': ['2016-07-04', '2017-07-04', '2018-07-04', '2019-07-04', '2020-07-04', '2021-07-04', '2022-07-04', '2023-07-04'],
            'Super Saturday': ['2016-12-24', '2017-12-23', '2018-12-22', '2019-12-21', '2020-12-19', '2021-12-18', '2022-12-24', '2023-12-23'],
            'New Year\'s Day': ['2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
        }
        self.populate_sales_df()
    
    def populate_sales_df(self):
        """
        Populates the sales DataFrame with sales events and their dates.
        """
        for sale_day, dates in self.us_sales_days.items():
            for date in dates:
                date = pd.to_datetime(date)
                self.sales_df.loc[date, 'events'] = sale_day
        self.sales_df.dropna(subset=['events'], inplace=True)
    
    def get_sales_df(self):
        """
        Retrieves the sales DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing sales events and their dates.
        """
        return self.sales_df

