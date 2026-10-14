"""
Feature Engineering and Data Preprocessing Module
Dataset: traffic.csv (48,120 rows across 4 highway junctions)
"""

import pandas as pd

# The 7 features extracted from traffic.csv for training
FEATURE_COLUMNS = [
    'Junction',
    'hour',
    'day_of_week',
    'day',
    'month',
    'year',
    'is_weekend'
]


def load_and_preprocess_dataset(filepath: str = 'traffic.csv') -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads traffic.csv and decomposes 'DateTime' into discrete temporal features.

    Returns:
        X (pd.DataFrame): 7 feature columns
        y (pd.Series): Target 'Vehicles' column
    """
    df = pd.read_csv(filepath)

    # Convert string timestamp to datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Extract temporal components
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['day'] = df['DateTime'].dt.day
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    X = df[FEATURE_COLUMNS]
    y = df['Vehicles']
    return X, y


def format_user_input(data: dict) -> pd.DataFrame:
    """
    Converts incoming user payload into a single-row DataFrame matching FEATURE_COLUMNS.
    """
    junction = int(data.get('junction', 1))

    if 'datetime' in data and data['datetime']:
        dt = pd.to_datetime(str(data['datetime']).replace('T', ' '))
        hour = dt.hour
        day_of_week = dt.dayofweek
        day = dt.day
        month = dt.month
        year = dt.year
        is_weekend = 1 if day_of_week >= 5 else 0
    else:
        hour = int(data.get('hour', 9))
        day_of_week = int(data.get('day_of_week', 0))
        day = int(data.get('day', 15))
        month = int(data.get('month', 10))
        year = int(data.get('year', 2026))
        is_weekend = 1 if day_of_week >= 5 else int(data.get('is_weekend', 0))

    row = {
        'Junction': junction,
        'hour': hour,
        'day_of_week': day_of_week,
        'day': day,
        'month': month,
        'year': year,
        'is_weekend': is_weekend
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]
