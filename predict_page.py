import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
from data_cache import load_data

pd.plotting.register_matplotlib_converters()

df = load_data()

avg_sales = df.groupby('date').agg({'bottles_sold': 'mean'}).reset_index()
avg_sales = avg_sales.set_index('date').to_period("D")
avg_sales['weekly_avg_sales'] = avg_sales['bottles_sold'].ewm(span=7, adjust=False).mean()

def show_predict_page():
    st.title("Predict Sales")

    st.write(
        """
    ### Predict Average Sales, Across All Stores, Categories
    """
    )

    days = st.slider("Predict", 0, 60, 30)
    # Retrieve trained model
    pickled_model = pickle.load(open('pickles/overall_sales.sav', 'rb'))
    pickled_dp = pickle.load(open('pickles/overall_dp.sav', 'rb'))

    X = pickled_dp.in_sample()
    y = avg_sales["bottles_sold"]
    y_pred = pd.Series(pickled_model.predict(X), index=y.index)
    X_fore = pickled_dp.out_of_sample(steps=days)
    y_fore = pd.Series(pickled_model.predict(X_fore), index=X_fore.index)

    # plot
    date_index = y.index.union(X_fore.index)

    chart_data = pd.DataFrame({
      'Average': y_pred,
      'Forecast': y_fore
    }, index=date_index)

    st.line_chart(chart_data)

   

    