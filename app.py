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

tab1, tab2, tab3 = st.tabs(["Explore & Analysis", "Algorithm Comparison", "Prediction"])

with tab1:
   show_explore_page()

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   show_predict_page()
