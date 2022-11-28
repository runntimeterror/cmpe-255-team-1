import streamlit as st
st.set_page_config(
    page_title="CMPE-255 Term Assignment",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://github.com/runntimeterror/cmpe-255-team-1/issues",
        'About': "Appliation developed for CMPE-255 term project."
    }
)

from predict_page import show_predict_page
from explore_page import show_explore_page
page = st.sidebar.selectbox("Explore Or Predict", ("Explore", "Predict"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()