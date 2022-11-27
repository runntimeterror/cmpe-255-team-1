import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

@st.cache
def load_data():
    df = pd.read_csv("data.csv")
    return df
df = load_data()

def show_explore_page():
    st.title("Explore stuff")

    st.write(
        """
    ### Stack Overflow Developer Survey 2020
    """
    )


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

    