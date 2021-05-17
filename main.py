import src.database.build_db as db
import src.weather as weather
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
import datetime
import numpy as np
import pickle
import h2o


rnd_forest_model = pickle.load(open("models/best_rf.pkl", 'rb'))

st.set_option('deprecation.showPyplotGlobalUse', False)

# My functions

if __name__ == '__main__':
    conn = db.connect_to_mysql()
    if conn:
        db.create_schemas(conn)

# load data
df_sales = pd.read_csv('data/raw_data/sales_clean.csv')

df_weather = weather.df_weather


# Other functions
def get_holidays(date_min, date_max):
    holidays = pd.read_sql_query(
        f"""
                SELECT
                *
                FROM Holidays
                WHERE
                date >= '{date_min}' and date <= '{date_max}'
                """, conn
    )
    holidays = holidays.fillna(value=np.nan)
    del holidays['id']
    holidays['date'] = pd.to_datetime(holidays['date'])
    return holidays

# Other functions


def prepare_data(df_weather, model):
    df_test = df_weather.copy()
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_test["day_of_week"] = df_test.date.dt.day_name()
    df_test["month_name"] = df_test.date.dt.month_name()
    df_test["day"] = df_test.date.dt.day
    df_test["year"] = df_test.date.dt.year

    df_placeholder = df_sales.drop(columns=['total_sales'])
    df_placeholder['date'] = pd.to_datetime(df_placeholder['date'])
    print(df_placeholder)

    df_test = df_test.append(df_placeholder)

    df_test['did_rain'] = df_test['type'].apply(
        lambda x: 1 if x == 'Rain' else 0)
    df_test['total_precip_mm'] = df_test['did_rain'].apply(
        lambda x: 0.5 if x == 1 else 0)

    date_min = df_test.date.min()
    date_max = df_test.date.max()
    df_holidays = get_holidays(date_min, date_max)

    df_merged = pd.merge(df_test, df_holidays, how="right", on="date")

    df_merged['is_closed'] = 0
    df_merged['is_lockdown'] = 0
    df_merged['is_curfew'] = 0

    df_merged = df_merged.set_index('date')
    df_merged['year'] = df_merged.year.astype('category')

    del df_merged['day']
    del df_merged['holiday_type']
    del df_merged['holiday_name']
    del df_merged['type']

    df_merged = pd.get_dummies(df_merged, dummy_na=True)
    df_merged = df_merged.iloc[-8:]
    del df_merged['day_type_nan']
    del df_merged['month_name_nan']
    del df_merged['day_of_week_nan']
    del df_merged['year_nan']

    df_merged['prev_sales'] = df_sales.total_sales.mean()
    df_merged['is_post_holiday'] = 0
    df_merged['is_pre_holiday'] = 0

    df_test = df_merged.copy()
    results = predict_data(df_test, model)
    return results


def predict_data(df_test, model):
    if model == 'forest':
        df_results = df_weather.copy()
        predictions = rnd_forest_model.predict(df_test)
        round_predicts = [round(num, 2) for num in predictions]

        print(round_predicts)

        df_results['Sales Prediction'] = pd.Series(
            round_predicts, index=df_results.index)

        return df_results
    elif model == "deep":
        df_results = df_weather.copy()
        h2o.init()
        saved_model = h2o.load_model(
            '/mnt/c/Users/lesto/Desktop/Ironhack/CityPlayForecast/models/deeplearning/DeepLearning_grid__2_AutoML_20210515_173143_model_1')
        stacked_test = df_test.copy()
        # Conversion into a H20 frame to train
        h2test = h2o.H2OFrame(stacked_test)
        predicted_price_h2 = saved_model.predict(
            h2test).as_data_frame()
        print(predicted_price_h2)
        df_results['Sales Prediction'] = predicted_price_h2.predict

        return df_results


forest_weekly_outlook = prepare_data(df_weather, "forest")
deep_learn_model = prepare_data(df_weather, "deep")

st.write("""
# CityPlay Sales Prediction App
This app predicts **sales for CityPlay, a bowling alley in Madrid**!
""")
st.write('---')

# sidebar
st.sidebar.header('Input Parameters')


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
st.text('Weekly Outlook (Random Forest Model)')
st.dataframe(forest_weekly_outlook)
st.text('Weekly Outlook (Deep Learning Model)')
st.dataframe(deep_learn_model)


df_sales.plot(x='date', y='total_sales', figsize=(20, 6))
plt.legend()
st.pyplot()
