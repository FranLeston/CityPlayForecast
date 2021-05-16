import src.database.build_db as db
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from dotenv import load_dotenv
import requests
import os
import pymysql
import sys
import json

import streamlit as st
import matplotlib as mpl

import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

# My functions

if __name__ == '__main__':
    conn = db.connect_to_mysql()
    if conn:
        db.create_schemas(conn)

# load data
df_sales = pd.read_csv('data/raw_data/sales_clean.csv')

st.write("""
# CityPlay Sales Prediction App
This app predicts **sales for CityPlay, a bowling alley in Madrid**!
""")
st.write('---')

# sidebar
st.sidebar.header('Specify Input Parameters')


def user_input_features():
    st.sidebar.date_input('Date input')
    holidays = ["Holiday", "Holiday-eve", "post-Holiday", "Normal day"]

    st.sidebar.radio("What Type of day is it?", holidays)
    st.sidebar.number_input('Do you happen to know the sales the day before? The default is the average.',
                            min_value=0.00, max_value=20000.00, value=2455.00, step=20.00, format=None, key=None)

    data = {}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# main
df_sales.plot(x='date', y='total_sales', figsize=(20, 6))
plt.legend()
st.pyplot()

