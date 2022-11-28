import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_cache import load_data
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import numpy as np
import pydeck as pdk

df = load_data()

daily_sales = df.groupby('date', as_index=False)['bottles_sold'].sum()
store_daily_sales = df.groupby(['vendor_name', 'date'], as_index=False)['bottles_sold'].sum()
item_daily_sales = df.groupby(['item_number', 'date'], as_index=False)['bottles_sold'].sum()

def point_to_geo(point):
  split = point.split(' ')
  long = float(split[0].replace('POINT(',''))
  lat = float(split[1].replace(')', ''))
  return pd.Series({'lat':lat, 'long':long})

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
        st.code("""
        df = load_data()
daily_sales = df.groupby('date', as_index=False)['bottles_sold'].sum()
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

    st.write(
        """
    #### Geographical Distribution of Sale
    """
    )

    sale_volume_by_stores = df.groupby('store_location', as_index=False)['sale_dollars'].sum()
    result = sale_volume_by_stores['store_location'].apply(point_to_geo)
    chart_data = sale_volume_by_stores.join(result)

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=41.8780,
            longitude=-93.0977,
            zoom=7,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
            'HexagonLayer',
            data=chart_data,
            get_position=['long', 'lat'],
            radius=1500,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            )
        ],
    ))

    with st.expander("See explanation"):
        st.write("""
            This uses the geographic coordinates of each store, and the sum of the total sale to plot.
        """)
        st.code("""
        sale_volume_by_stores = df.groupby('store_location', as_index=False)['sale_dollars'].sum()
result = sale_volume_by_stores['store_location'].apply(point_to_geo)
chart_data = sale_volume_by_stores.join(result)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=41.8780,
        longitude=-93.0977,
        zoom=7,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
        'HexagonLayer',
        data=chart_data,
        get_position=['long', 'lat'],
        radius=1500,
        elevation_scale=4,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        )
    ],
))
        """)

    