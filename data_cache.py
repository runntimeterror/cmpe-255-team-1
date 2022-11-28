import streamlit as st
import pandas as pd

@st.cache
def load_data():
    df = pd.read_csv("data.csv")
    df_date_sales = date_features(df)
    return df_date_sales


def date_features(df):
    # Date Features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['weekofyear'] = df.date.dt.isocalendar().week
    
    return df
