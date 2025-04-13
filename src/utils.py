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

def display_article(row):
    # Display news articles
    is_adverse = row['category'] not in ['General Financial News', 'Non Financial News']
    news_tag = "🔴" if is_adverse else "🔵"
    # for _, row in df.iterrows():
    with st.expander(f"{news_tag} {row['title']}"):
        st.markdown(f"**Published**: {row['date']}")
        st.markdown(f"**URL**: [Link]({row['url']})")
        st.markdown(f"**Snippet**: {row['snippet']}")
        if is_adverse:
            st.markdown(f"**Predicted Category**: :red-background[{row['category']}]")
        else:
            st.markdown(f"**Predicted Category**: :blue-background[{row['category']}]")
        st.markdown(f"**Justification**: `{row['justification']}`")
        if is_adverse:
            st.markdown(f"**📌Adverse Relevance**: :red[{row['adverse_relevance']}]")

        st.markdown("**Entities Extracted:**")
        entities = row['entities']
        entities_flag = 0
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
            entities_flag = 1
            st.markdown(f"**🧠 {entity_type.replace('_', ' ').title()}**")
            for entity in entity_list:
                if isinstance(entity, dict):
                    name = entity.get("entity_name")
                    variations = entity.get("variations", [])
                    variation_str = ", ".join([v for v in variations if v.lower() != name.lower()])
                    if variation_str:
                        st.markdown(f"- **{name}**  \
                        _Also referred to as:_ {variation_str}")
                    else:
                        st.markdown(f"- **{name}** \
                                    _No other reference found_")
        if not entities_flag:
            st.markdown("_No entities found_")

