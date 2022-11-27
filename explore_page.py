import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_cache import load_data
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

df = load_data()

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
df_date_sales = date_features(df)

daily_sales = df_date_sales.groupby('date', as_index=False)['bottles_sold'].sum()
store_daily_sales = df_date_sales.groupby(['store_number', 'date'], as_index=False)['bottles_sold'].sum()
item_daily_sales = df_date_sales.groupby(['item_number', 'date'], as_index=False)['bottles_sold'].sum()

def show_explore_page():
    st.title("Data Exploration")

    st.write(
        """
    ### Average Daily Sales
    """
    )

    #Overall daily sales

    st.line_chart(data=daily_sales, x='date', y='bottles_sold')

    st.write("""#### Number of Data from different countries""")

    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    
    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    