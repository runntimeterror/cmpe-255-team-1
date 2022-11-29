import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_cache import load_data
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import numpy as np
import pydeck as pdk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression 
# import xgboost as xg 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from time import time
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LassoLars, Lars, SGDRegressor
from sklearn.svm import NuSVR
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess



def algorithm_loop(X_train, X_test, y_train, y_test, X_lookahead):
    names = ["NuSVR","Lars", "Lasso Lars", "Linear Regression"]

    classifiers = [
    NuSVR(),
    Lars(),
    LassoLars(alpha=0.01),
    # xg.XGBRegressor(objective ='reg:linear',n_estimators = 10, seed = 123),
    LinearRegression(fit_intercept=False)
    ]

    max_score = 0.0
    max_class = ''

    clf_best = ''
    
    # Iterate over classifiers
    for name, clf in zip(names, classifiers):
      start_time = time()
      model = clf.fit(X_train, y_train)

      score = 100.0 * clf.score(X_test, y_test)
      if score > max_score:
          clf_best = clf
          max_score = score
          max_class = name


      y_pred = pd.Series(model.predict(X_train), index=y_train.index)
      y_fore = pd.Series(model.predict(X_lookahead), index=X_lookahead.index)
      
      fig, ax = plt.subplots()
      pt = y_train.plot(color='0.25', style='.', title='Sales- Forecast (' + name +')', figsize=[20,10])
      pt = y_pred.plot(ax=pt, label="Average Sales")
      pt = y_fore.plot(ax=pt, label="Forecast", color='C3')
      pt = y_test.plot(ax=pt, label="Actual", color='C4')

      _ = pt.legend()
    #   plt.figure()
      st.pyplot(fig)

    return max_class, clf_best


df = load_data()
avg_sales = df.groupby('date').agg({'bottles_sold': 'mean'}).reset_index()
avg_sales = avg_sales.set_index('date').to_period("D")
avg_sales['weekly_avg_sales'] = avg_sales['bottles_sold'].ewm(span=7, adjust=False).mean()

# daily_sales = df.groupby('date', as_index=False)['bottles_sold'].sum()
# store_daily_sales = df.groupby(['vendor_name', 'date'], as_index=False)['bottles_sold'].sum()
# item_daily_sales = df.groupby(['item_number', 'date'], as_index=False)['bottles_sold'].sum()

def predict_compare(split_date, length):
  before_oct = avg_sales[avg_sales.index < split_date]
  oct_forward = avg_sales[avg_sales.index >= split_date]

  y_train = before_oct["bottles_sold"]
  y_test = oct_forward["bottles_sold"]
  print("ytest_len", len(y_test))

  fourier = CalendarFourier(freq='M', order=4)

  dp = DeterministicProcess(
      index=before_oct.index,
      constant=True,   # dummy feature for bias (y-intercept)
      order=1,         # trend ( order 1 means linear)
      seasonal=True,   # weekly seasonality (indicators)
      additional_terms=[fourier], # annual seasonality
      drop=True,       # drop terms to avoid collinearity
  )

  X_train = dp.in_sample()

  dp_2 = DeterministicProcess(
      index=oct_forward.index,
      constant=True,   # dummy feature for bias (y-intercept)
      order=1,         # trend ( order 1 means linear)
      seasonal=True,   # weekly seasonality (indicators)
      additional_terms=[fourier], # annual seasonality
      drop=True,       # drop terms to avoid collinearity
  )

  X_test = dp_2.in_sample()

  print(len(X_test))
  X_lookahead = dp.out_of_sample(length)

  algorithm_loop(X_train, X_test, y_train, y_test, X_lookahead)


def show_algorithm_comparison_page():
    st.title("Algorithm Comparison")

    st.write(
        """
    ### Comparing Based on Hold Out and Algorithm
    ****Holdout Periods (Training on Data Up To...)****
    1. 09/01/2022
    2. 09/15/2022
    3. 10/01/2022
    4. 10/10/2022
    ****Algorithms****
    1. "NuSVR"
    2. "Lars"
    3. "Lasso Lars"
    4. "Linear Regression"
    #### Conclusions
    Looking at the charts, the algorithm that makes the best predictions 
    is either LARS, Lasso LARS, or Linear Regression.

    With regards to the holdout data, the all three of the holdout periods
    seem to perform similarly, where the model tends to avoid the outliers
    and the predictions from the model does the same.
    """
    )

    st.write(
        """
    #### Training Data: training with data up to 9/01/2022 
        """
    )


    # with st.expander("See explanation"):
    #     st.write("""
    #         The chart above shows the average daily sales (bottles sold) across all stores.
    #     """)

    #Overall daily sales
    predict_compare('09/01/2022', 60)

    st.write(
        """
    #### Training Data: training with data up to 9/15/2022 
        """
    )
    predict_compare('09/15/2022', 46)
    
    st.write(
        """
    #### Training Data: training with data up to 10/01/2022 
        """
    )
    predict_compare('10/01/2022', 31)

    st.write(
        """
    #### Training Data: training with data up to 10/10/2022 
        """
    )
    predict_compare('10/10/2022', 22)

#     st.line_chart(data=daily_sales, x='date', y='bottles_sold')

#     with st.expander("See explanation"):
#         st.write("""
#             The chart above shows the average daily sales (bottles sold) across all stores.
#         """)
#         st.code("""
#         df = load_data()
# daily_sales = df.groupby('date', as_index=False)['bottles_sold'].sum()
#         """)

#     st.write("""#### Daily Sale Trends """)
#     #daily sales by distribution center
#     data_grouped_day = df.groupby(['dayofweek']).mean()['bottles_sold']
#     labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#     fig, ax = plt.subplots()
#     ax.bar(labels, data_grouped_day, 0.35)
    
#     st.pyplot(fig)

    

#     st.write(
#         """
#     #### Daily Sales By Vendor
#     """
#     )
#     store_daily_sales_sc = []
#     for store in store_daily_sales['vendor_name'].unique():
#         current_store_daily_sales = store_daily_sales[(store_daily_sales['vendor_name'] == store)]
#         store_daily_sales_sc.append(go.Scatter(x=current_store_daily_sales['date'], y=current_store_daily_sales['bottles_sold'], name=('%s' % store)))

#     layout = go.Layout(title='Vendor daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
#     fig2 = go.Figure(data=store_daily_sales_sc, layout=layout)
#     st.plotly_chart(fig2)
    
#     st.write(
#         """
#     #### Sales by Category
#     """
#     )

#     # Count all sales by category and plot
#     sales_by_category = df.groupby(['category_name']).agg({'bottles_sold':'sum'})
#     sales_by_category = sales_by_category.sort_index(ascending=[True])
#     st.bar_chart(sales_by_category, height=800)

#     st.write(
#         """
#     #### Geographical Distribution of Sale
#     """
#     )

#     sale_volume_by_stores = df.groupby('store_location', as_index=False)['sale_dollars'].sum()
#     result = sale_volume_by_stores['store_location'].apply(point_to_geo)
#     chart_data = sale_volume_by_stores.join(result)

#     st.pydeck_chart(pdk.Deck(
#         map_style=None,
#         initial_view_state=pdk.ViewState(
#             latitude=41.8780,
#             longitude=-93.0977,
#             zoom=7,
#             pitch=50,
#         ),
#         layers=[
#             pdk.Layer(
#             'HexagonLayer',
#             data=chart_data,
#             get_position=['long', 'lat'],
#             radius=1500,
#             elevation_scale=4,
#             elevation_range=[0, 1000],
#             pickable=True,
#             extruded=True,
#             )
#         ],
#     ))

#     with st.expander("See explanation"):
#         st.write("""
#             This uses the geographic coordinates of each store, and the sum of the total sale to plot.
#         """)
#         st.code("""
#         sale_volume_by_stores = df.groupby('store_location', as_index=False)['sale_dollars'].sum()
# result = sale_volume_by_stores['store_location'].apply(point_to_geo)
# chart_data = sale_volume_by_stores.join(result)

# st.pydeck_chart(pdk.Deck(
#     map_style=None,
#     initial_view_state=pdk.ViewState(
#         latitude=41.8780,
#         longitude=-93.0977,
#         zoom=7,
#         pitch=50,
#     ),
#     layers=[
#         pdk.Layer(
#         'HexagonLayer',
#         data=chart_data,
#         get_position=['long', 'lat'],
#         radius=1500,
#         elevation_scale=4,
#         elevation_range=[0, 1000],
#         pickable=True,
#         extruded=True,
#         )
#     ],
# ))
#         """)

    