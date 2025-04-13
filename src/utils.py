import re
import streamlit as st
import pandas as pd

from collections import Counter
from datetime import datetime, timedelta
from dateutil import parser
import plotly.express as px


def parse_date(date_str):
    """
    Parse arbitrary date format into YYYY-mm-dd.
    """
    date_str = date_str.strip().lower()

    # Handle relative dates like "3 days ago", "2 weeks ago"
    relative_match = re.match(r'(\d+)\s+(day|week|month|year)s?\s+ago', date_str)
    if relative_match:
        quantity = int(relative_match.group(1))
        unit = relative_match.group(2)

        if unit == 'day':
            delta = timedelta(days=quantity)
        elif unit == 'week':
            delta = timedelta(weeks=quantity)
        elif unit == 'month':
            # Approximate 1 month as 30 days
            delta = timedelta(days=30 * quantity)
        elif unit == 'year':
            # Approximate 1 year as 365 days
            delta = timedelta(days=365 * quantity)

        parsed_date = datetime.now() - delta
    else:
        try:
            parsed_date = parser.parse(date_str)
        except ValueError:
            return None

    return parsed_date.strftime('%Y-%m-%d')

def patch_dataframe_date(df, group_column, date_column="date"):
    # Patch time series dataframe with full time range
    all_dates = pd.date_range(df[date_column].min(), df[date_column].max(), freq="D")
    all_categories = df[group_column].unique()
    complete_index = pd.MultiIndex.from_product([all_dates, all_categories], names=[date_column,group_column])
    complete_df = pd.DataFrame(index=complete_index).reset_index()
    complete_df[date_column] = pd.to_datetime(complete_df[date_column]).dt.date
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    df_full = complete_df.merge(df, on=[date_column, group_column], how="left").fillna(0)
    return df_full

