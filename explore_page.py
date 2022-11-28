import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_cache import load_data
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

df = load_data()

daily_sales = df.groupby('date', as_index=False)['bottles_sold'].sum()
store_daily_sales = df.groupby(['vendor_name', 'date'], as_index=False)['bottles_sold'].sum()
item_daily_sales = df.groupby(['item_number', 'date'], as_index=False)['bottles_sold'].sum()

def show_explore_page():
    st.title("Data Exploration")

    st.write(
        """
    ### Average Daily Sales
    """
    )

    #Overall daily sales

    st.line_chart(data=daily_sales, x='date', y='bottles_sold')

    with st.expander("See explanation"):
        st.write("""
            The chart above shows the average daily sales (bottles sold) across all stores.
        """)

    st.write("""#### Daily Sale Trends """)
    #daily sales by distribution center
    data_grouped_day = df.groupby(['dayofweek']).mean()['bottles_sold']
    labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig, ax = plt.subplots()
    ax.bar(labels, data_grouped_day, 0.35)
    
    st.pyplot(fig)

    

    st.write(
        """
    #### Daily Sales By Vendor
    """
    )
    store_daily_sales_sc = []
    for store in store_daily_sales['vendor_name'].unique():
        current_store_daily_sales = store_daily_sales[(store_daily_sales['vendor_name'] == store)]
        store_daily_sales_sc.append(go.Scatter(x=current_store_daily_sales['date'], y=current_store_daily_sales['bottles_sold'], name=('%s' % store)))

    layout = go.Layout(title='Vendor daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
    fig2 = go.Figure(data=store_daily_sales_sc, layout=layout)
    st.plotly_chart(fig2)
    
    st.write(
        """
    #### Sales by Category
    """
    )

    # Count all sales by category and plot
    sales_by_category = df.groupby(['category_name']).agg({'bottles_sold':'sum'})
    sales_by_category = sales_by_category.sort_index(ascending=[True])
    st.bar_chart(sales_by_category, height=800)

    