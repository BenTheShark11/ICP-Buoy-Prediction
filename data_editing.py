import pandas as pd
import numpy as np


def hybrid_impute(df, max_gap=3):
    """
    this function fills in a dataframe missing points with either interpolated datapoints or median-filled
    based on whether the gap is larger than 3 points, if it 3 or less, it interpolates, if it is greater than 3
    it fills it with the median of the column
    :param df: dataframe you are imputing
    :param max_gap: the max gap between data points used for interpolation vs median filling
    :return: returns the imputed dataframe fully filled in
    """
    result = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        # First, interpolate all missing values (limit restricts to small gaps)
        result[col] = df[col].interpolate('time', limit=max_gap, limit_direction='both')

        # For remaining NaNs (those in large gaps), fill with median
        median = df[col].median()
        result[col] = result[col].fillna(median)
    return result


def process_year(file_path, max_gap=3):
    """
    This function takes in a dataset and changes it to an hourly version that has been cleaned and interpolated
    :param file_path: the path of the text file for the year's data you are using
    :param max_gap: the max gap between data points used for interpolation vs median filling
    :return: the dataset fully cleaned, interpolated, and formatted to an hourly time step
    """

    # create a list of all the column names in order
    colnames = [
        'YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST',
        'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP',
        'DEWP', 'VIS', 'TIDE'
    ]

    # read the data, skipping the first two comment/header lines
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        skiprows=2,
        names=colnames
    )

    # remove columns with no real data
    cols_to_remove = ['WVHT', 'DPD', 'APD', 'MWD', 'DEWP', 'VIS', 'TIDE']
    df = df.drop(columns=cols_to_remove)

    # Convert all columns (except date parts) to numeric, setting all non numeric values to NaN
    for col in df.columns[5:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # the dataset sets all missing data to 9s, so this is setting all the values that it could possibly be for the missing values
    missing_map = {
        'WDIR': 999,
        'WSPD': [99.0, 99.00],
        'GST': [99.0, 99.00],
        'PRES': [999, 9999],
        'ATMP': [999.0, 99.0],
        'WTMP': [999.0, 99.0],
    }

    # sets all missing values from 9s to NaN
    for col, missing_vals in missing_map.items():
        df[col] = df[col].replace(missing_vals, np.nan)

    # there are some pressure values that are way too high but not exactly 9s, so they are being replaced with NaNs
    df.loc[df['PRES'] > 1100, 'PRES'] = np.nan

    # rename the columns to word format so that the datetime pandas can read it easily
    df = df.rename(columns={
        'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'
    })

    # create a datetime index of the original dataset
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    df = df.set_index('datetime').sort_index()
    df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute'])

    # insert rows into the dataset for each 6 minute interval missing
    # all new rows are set entirely to NaN and will be imputed later
    full_idx = pd.date_range(start=df.index.min(),
                             end=df.index.max(),
                             freq='6T')
    df = df.reindex(full_idx)

    # impute the missing data that was set to NaN
    df_imputed = hybrid_impute(df, max_gap=max_gap)

    # sets up which columns should be maxed or averaged per hour, all are averaged except peak wind gust which is the max value per hour
    columns = list(df_imputed.columns)
    agg_dict = {col: ('max' if col == 'GST' else 'mean') for col in columns}

    # changes the time interval from 6-minute to hourly
    df_hourly = df_imputed.resample('1H').agg(agg_dict)

    # extract month, day, hour from index (which is DatetimeIndex)
    df_hourly['month'] = df_hourly.index.month
    df_hourly['day'] = df_hourly.index.day
    df_hourly['hour'] = df_hourly.index.hour

    # I want to include the datetime data in the model's input to show it trends based on when in the season
    # and when in the day the data was recorded
    # most likely day of month is not necessary here, but it is included just to give more detail on when in the season it is

    # cyclical encoding for month
    df_hourly['month_sin'] = np.sin(2 * np.pi * df_hourly['month'] / 12)
    df_hourly['month_cos'] = np.cos(2 * np.pi * df_hourly['month'] / 12)

    # cyclical encoding for day (assuming 31 days max in month), some error here because of the assumption of 31 days per month
    df_hourly['day_sin'] = np.sin(2 * np.pi * df_hourly['day'] / 31)
    df_hourly['day_cos'] = np.cos(2 * np.pi * df_hourly['day'] / 31)

    # cyclical encoding for hour
    df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)
    df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)

    # drop the raw time columns
    df_hourly = df_hourly.drop(['month', 'day', 'hour'], axis=1)

    # do the same process for the wind direction
    df_hourly['WDIR_sin'] = np.sin(2 * np.pi * df_hourly['WDIR'] / 360)
    df_hourly['WDIR_cos'] = np.cos(2 * np.pi * df_hourly['WDIR'] / 360)
    df_hourly = df_hourly.drop(columns=['WDIR'])

    return df_hourly


df_dataset_2022 = process_year(
    r'C:\Users\HP\PycharmProjects\deepLearning\Neural Networks Project\Data Manipulation\Raw Data\Clearwater_2022.txt')
df_dataset_2023 = process_year(
    r'C:\Users\HP\PycharmProjects\deepLearning\Neural Networks Project\Data Manipulation\Raw Data\Clearwater_2023.txt')
print('debugging')

# save the dataframe as a csv file
#df_dataset_2022.to_csv('csv_dataset_2022.csv')

# Concatenate your two dataframes
df_all = pd.concat([df_dataset_2022, df_dataset_2023])

# make sure the datasets are in chronological order based on their datetime indexes
df_all = df_all.sort_index()

df_all.to_csv('csv_dataset_all.csv')