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
import datetime
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import h2o

h2o.init()

rnd_forest_model = pickle.load(open("models/best_rf.pkl", 'rb'))
st.set_page_config(layout="wide")
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

        df_test = df_test.append(df_placeholder)

        conditions = ["Rain", "Thunderstorm", "Drizzle", "Snow"]
        df_test['did_rain'] = df_test['type'].apply(
            lambda x: 1 if x in conditions else 0)
        df_test['total_precip_mm'] = df_test['did_rain'].apply(
            lambda x: 8 if x == 1 else 0)

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
            # df_results = df_weather.copy()
            df_results = pd.DataFrame()
            predictions = rnd_forest_model.predict(df_test)
            round_predicts = [round(num, 2) for num in predictions]

            df_results = pd.DataFrame(round_predicts, columns=[
                'predict'])

            # df_results['Sales Prediction'] = pd.Series(round_predicts, index=df_results.index)

            return df_results
        elif model == "deep":
            # df_results = df_weather.copy()
            # h2o.init()
            saved_model = h2o.load_model(
                'models/deeplearning/DeepLearning_grid__2_AutoML_20210515_173143_model_1')
            stacked_test = df_test.copy()
            # Conversion into a H20 frame to train
            h2test = h2o.H2OFrame(stacked_test)
            predicted_price_h2 = saved_model.predict(
                h2test).as_data_frame()

            df_results = predicted_price_h2.copy()
            # df_results['Sales Prediction'] = predicted_price_h2.predict

            return df_results
        elif model == "stacked":
            # df_results = df_weather.copy()
            # h2o.init()
            saved_model = h2o.load_model(
                'models/autostacked/StackedEnsemble_AllModels_AutoML_20210517_174810')
            stacked_test = df_test.copy()
            # Conversion into a H20 frame to train
            h2test = h2o.H2OFrame(stacked_test)
            predicted_price_h2 = saved_model.predict(
                h2test).as_data_frame()

            df_results = predicted_price_h2.copy()
            # df_results['Sales Prediction'] = predicted_price_h2.predict

            return df_results

    h20_stacked_model = prepare_data(df_weather, "stacked")
    forest_weekly_outlook = prepare_data(df_weather, "forest")
    deep_learn_model = prepare_data(df_weather, "deep")

    df_main = df_weather.copy()
    df_main.rename(columns={"type": "Weather",
                            "average_temp": "Temp"}, inplace=True)
    df_main['RndForest Sales'] = forest_weekly_outlook['predict']
    df_main['DeepLearn Sales'] = deep_learn_model['predict']
    df_main['h2O Sales'] = h20_stacked_model['predict']
    df_main['date'] = pd.to_datetime(df_main['date'])
    df_main['date'] = df_main['date'].dt.strftime('%d/%m/%Y')

    df_main['AVG Prediction'] = df_main[[
        'DeepLearn Sales', 'h2O Sales', 'RndForest Sales']].mean(axis=1)

    def get_employees(x):
        if x < 1000:
            return 3
        elif x >= 1000 and x < 2000:
            return 4
        elif x >= 2000 and x < 4000:
            return 5
        elif x >= 4000 and x < 6000:
            return 6
        elif x >= 6000 and x < 8000:
            return 7
        elif x >= 8000 and x < 10000:
            return 8
        elif x >= 10000:
            return 10

    df_main['Employees Needed'] = df_main['AVG Prediction'].apply(
        lambda x: get_employees(x))

    st.image('images/logo_large.png', width=300)
    st.write("""
    # CityPlay Sales Prediction App
    This app predicts **sales for CityPlay, a bowling alley in Madrid**!
    """)
    st.write('---')

    # sidebar
    st.sidebar.header('Choose Input Params')

    def user_input_features():
        today = datetime.date.today()
        week_from_now = today + datetime.timedelta(days=8)

        start_date = st.sidebar.date_input('Date input', week_from_now)
        holidays = ["Normal day", "Holiday", "Holiday-eve", "post-Holiday"]
        holiday_choice = st.sidebar.radio("What Type of day is it?", holidays)
        prev_sales_value = st.sidebar.number_input('Do you happen to know the sales the day before? The default is the average.',
                                                   min_value=0.00, max_value=20000.00, value=2455.00, step=20.00, format=None, key=None)

        temp = st.sidebar.slider('Temperature', value=17,
                                 min_value=-15, max_value=48)
        will_it_rain = [False, True]
        did_rain_value = st.sidebar.radio(
            "Predict Rain/Snow (People will stay home it rains a lot!)", will_it_rain)
        if did_rain_value:
            mm = st.sidebar.slider('RainFall in millimeters', value=8.5,
                                   min_value=0.5, max_value=29.5, step=0.5)
        else:
            mm = 0

        data_list = [start_date]

        df_test_day = pd.DataFrame([data_list])
        df_test_day.columns = ['date']

        df_test_day['date'] = pd.to_datetime(df_test_day['date'])
        df_test_day2 = df_test_day.copy()

        day_of_week = str(df_test_day.date.dt.day_name()).split()[1]
        print("DAY:", day_of_week)
        weekend = ["Saturday", "Sunday"]

        df_test_day["day_of_week"] = df_test_day.date.dt.day_name()
        df_test_day["month_name"] = df_test_day.date.dt.month_name()
        df_test_day["day"] = df_test_day.date.dt.day
        df_test_day["year"] = df_test_day.date.dt.year

        df_placeholder2 = df_sales.drop(columns=['total_sales'])
        df_placeholder2['date'] = pd.to_datetime(df_placeholder2['date'])

        df_test_day = df_test_day.append(df_placeholder2)
        df_test_day['did_rain'] = 1 if did_rain_value == True else 0
        df_test_day['total_precip_mm'] = mm

        df_test_day['day_type_domingo'] = df_test_day['day_of_week'].apply(
            lambda x: 1 if x == "Sunday" else 0)
        df_test_day['day_type_sábado'] = df_test_day['day_of_week'].apply(
            lambda x: 1 if x == "Saturday" else 0)
        df_test_day['day_type_festivo'] = 1 if holiday_choice == "Holiday" else 0

        if holiday_choice == "Normal day":
            if day_of_week not in weekend:
                df_test_day['day_type_laborable'] = 1
            else:
                df_test_day['day_type_laborable'] = 0
        elif holiday_choice == "Holiday":
            if day_of_week in weekend:
                df_test_day['day_type_sábado'] = 0
                df_test_day['day_type_domingo'] = 0
            else:
                df_test_day['day_type_laborable'] = 0

        df_test_day['is_post_holiday'] = 1 if holiday_choice == "post-Holiday" else 0
        df_test_day['is_pre_holiday'] = 1 if holiday_choice == "Holiday-eve" else 0
        df_test_day['average_temp'] = temp
        df_test_day['is_closed'] = 0
        df_test_day['is_lockdown'] = 0
        df_test_day['is_curfew'] = 0
        df_test_day['prev_sales'] = prev_sales_value

        df_test_day = df_test_day.set_index('date')
        df_test_day['year'] = df_test_day.year.astype('category')

        del df_test_day['day']

        df_test_day = pd.get_dummies(df_test_day, dummy_na=True)

        df_test_day = df_test_day.iloc[0:1]
        del df_test_day['month_name_nan']
        del df_test_day['day_of_week_nan']
        del df_test_day['year_nan']

        df_test2 = df_test_day.copy()

        df_test2.to_csv("sampletesting.csv", index=True)

        res_stacked_df = predict_data(df_test2, "stacked")
        res_deep_df = predict_data(df_test2, "deep")

        df_test_day2['date'] = pd.to_datetime(df_test_day2['date'])
        df_test_day2['date'] = df_test_day2['date'].dt.strftime('%d/%m/%Y')

        df_test_day2['DeepLearn Sales'] = res_deep_df['predict']
        df_test_day2['h2O Sales'] = res_stacked_df['predict']

        df_test_day2['AVG Prediction'] = df_test_day2[[
            'DeepLearn Sales', 'h2O Sales']].mean(axis=1)

        def get_employees(x):
            if x < 1000:
                return 3
            elif x >= 1000 and x < 2000:
                return 4
            elif x >= 2000 and x < 4000:
                return 5
            elif x >= 4000 and x < 6000:
                return 6
            elif x >= 6000 and x < 8000:
                return 7
            elif x >= 8000 and x < 10000:
                return 8
            elif x >= 10000:
                return 10

        df_test_day2['Employees Needed'] = df_test_day2['AVG Prediction'].apply(
            lambda x: get_employees(x))
        return df_test_day2

    results_df = user_input_features()

    # main

    st.text('Your Query:')
    st.dataframe(results_df.style.format(
        {'h2O Sales': '{:.2f}', 'DeepLearn Sales': '{:.2f}', 'AVG Prediction': '{:.2f}'}))

    st.text('Weekly Outlook')
    st.dataframe(df_main.style.format(
        {'Temp': '{:.1f}', 'RndForest Sales': '{:.2f}', 'h2O Sales': '{:.2f}', 'DeepLearn Sales': '{:.2f}', 'AVG Prediction': '{:.2f}'}))

    # VISUALS
    df_graphics = pd.read_csv('data/db_load_files/clean_data.csv')

    chart_list = ["Time Series of Sales", "Sales per Day of the week",
                  "Sales per Month", "Top Holidays", "Top Days"]
    selection = st.selectbox('Charts / Information', chart_list)

    # 1
    if selection == "Time Series of Sales":
        fig = px.line(df_graphics, x="date", y="total_sales",
                      title='Daily Sales 21/09/2021 - 11/05/2021')
        fig.data[0].line.color = '#00cb56'
        st.plotly_chart(fig, use_container_width=True)
    elif selection == "Sales per Day of the week":
        # 2
        fig2 = px.bar(df_graphics, x="year", color="day_of_week",
                      y='total_sales',
                      title="Sales / Day of the week",
                      barmode='group',
                      )
        st.plotly_chart(fig2, use_container_width=True)
    elif selection == "Sales per Month":
        # 3
        fig3 = px.bar(df_graphics, x="year", color="month_name",
                      y='total_sales',
                      title="Sales / Month",
                      barmode='group',
                      )
        st.plotly_chart(fig3, use_container_width=True)
    elif selection == "Top Holidays":
        # 4
        list_holidays = df_graphics.groupby('holiday_name').mean().sort_values(
            by=['total_sales'], ascending=False)

        list_holidays = list_holidays[[
            'total_sales', 'average_temp', 'total_precip_mm']]
        list_holidays.rename(columns={'total_sales': 'AVG Sales',
                                      'average_temp': 'AVG Temperature', 'total_precip_mm': "AVG Rainfall mm"}, inplace=True)

        st.text('Busiest Holidays')
        st.dataframe(list_holidays.style.format(
            {'AVG Sales': '{:.2f}', 'AVG Temperature': '{:.1f}', 'AVG Rainfall mm': '{:.1f}'}))
    elif selection == "Top Days":
        # 5
        list_ranking = df_graphics.sort_values(
            by=['total_sales'], ascending=False)

        list_ranking = list_ranking[[
            'date', 'total_sales', 'average_temp', 'total_precip_mm']]
        list_ranking.rename(columns={'total_sales': 'Sales',
                                     'average_temp': 'Temperature', 'total_precip_mm': "Rainfall mm"}, inplace=True)

        st.text('Top Sales in on day')
        st.dataframe(list_ranking.style.format(
            {'Sales': '{:.2f}', 'Temperature': '{:.1f}', 'Rainfall mm': '{:.1f}'}))
