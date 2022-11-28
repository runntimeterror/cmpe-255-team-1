import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
from data_cache import load_data
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression


fourier = CalendarFourier(freq='M', order=4)

pd.plotting.register_matplotlib_converters()

df = load_data()

avg_sales = df.groupby('date').agg({'bottles_sold': 'mean'}).reset_index()
avg_sales = avg_sales.set_index('date').to_period("D")
avg_sales['weekly_avg_sales'] = avg_sales['bottles_sold'].ewm(span=7, adjust=False).mean()

# generic function using Linear Regression
def predict_category(df_category, days):
  avg_sales_cat = df_category.groupby('date').agg({'bottles_sold': 'mean'}).reset_index()
  avg_sales_cat = avg_sales_cat.set_index('date').to_period("D")
  avg_sales_cat['weekly_avg_sales'] = avg_sales_cat['bottles_sold'].ewm(span=7, adjust=False).mean()
  dp_cat = DeterministicProcess(
      index=avg_sales_cat.index,
      constant=True,   # dummy feature for bias (y-intercept)
      order=1,         # trend ( order 1 means linear)
      seasonal=True,   # weekly seasonality (indicators)
      additional_terms=[fourier], # annual seasonality
      drop=True,       # drop terms to avoid collinearity
  )

  X_cat = dp_cat.in_sample()
  y_cat = avg_sales_cat["bottles_sold"]

  model_cat = LinearRegression(fit_intercept=False)
  model_cat.fit(X_cat, y_cat)

  y_cat_pred = pd.Series(model_cat.predict(X_cat), index=y_cat.index)
  X_cat_fore = dp_cat.out_of_sample(steps=days)
  y_cat_fore = pd.Series(model_cat.predict(X_cat_fore), index=X_cat_fore.index)

  # plot
  date_index = y_cat.index.union(X_cat_fore.index).astype(str)

  chart_data = pd.DataFrame({
    'Average': y_cat_pred,
    'Forecast': y_cat_fore
  }, index=date_index)

  return chart_data


def show_predict_page():
  st.title("Predict Sales")

  st.write(
      """
  ### Predict Average Sales, Across All Stores, Categories
  """
  )

  days = st.slider("Days into the future", 0, 60, 30)
  # Retrieve trained model
  pickled_model = pickle.load(open('pickles/overall_sales.sav', 'rb'))
  pickled_dp = pickle.load(open('pickles/overall_dp.sav', 'rb'))

  X = pickled_dp.in_sample()
  y = avg_sales["bottles_sold"]
  y_pred = pd.Series(pickled_model.predict(X), index=y.index)
  X_fore = pickled_dp.out_of_sample(steps=days)
  y_fore = pd.Series(pickled_model.predict(X_fore), index=X_fore.index)

  # plot
  date_index = y.index.union(X_fore.index).astype(str)

  chart_data = pd.DataFrame({
    'Average': y_pred,
    'Forecast': y_fore
  }, index=date_index)

  st.line_chart(chart_data)

  with st.expander("See explanation"):
        st.code("""
    # Retrieve trained model
  pickled_model = pickle.load(open('pickles/overall_sales.sav', 'rb'))
  pickled_dp = pickle.load(open('pickles/overall_dp.sav', 'rb'))

  # Retrieve X
  X = pickled_dp.in_sample()
  y = avg_sales["bottles_sold"]
  y_pred = pd.Series(pickled_model.predict(X), index=y.index)
  X_fore = pickled_dp.out_of_sample(steps=days)
  y_fore = pd.Series(pickled_model.predict(X_fore), index=X_fore.index)

  # plot
  date_index = y.index.union(X_fore.index).astype(str)

  chart_data = pd.DataFrame({
    'Average': y_pred,
    'Forecast': y_fore
  }, index=date_index)

  st.line_chart(chart_data)
        """)


  st.write(
        """
    #### Filter by Category & Vendor
    """
    )
  category_data = df['category_name'].unique()
  vendor_data = df['vendor_name'].unique()
  col1, col2 = st.columns([1, 1])

  category = col1.selectbox(
    'Select Category',
    category_data)

  vendor = col2.selectbox(
    'Select Vendor',
    vendor_data)

  df_category_vendor = df[(df['category_name']==category) & (df['vendor_name']==vendor)]
  if df_category_vendor.empty:
    st.caption('DataFrame is empty! Please change selection')
  else:
    chart_data_cat = predict_category(df_category_vendor, days)
    st.line_chart(chart_data_cat)
 
  with st.expander("See explanation"):
        st.code("""
    df_category_vendor = df[(df['category_name']==category) & (df['vendor_name']==vendor)]
  if df_category_vendor.empty:
    st.caption('DataFrame is empty! Please change selection')
  else:
    chart_data_cat = predict_category(df_category_vendor, days)
    st.line_chart(chart_data_cat)
        """)