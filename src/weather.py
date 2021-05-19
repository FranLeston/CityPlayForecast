import requests
import json
import pandas as pd
import os

from dotenv import load_dotenv
from datetime import datetime


load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
lat = 40.4638
lon = -3.6372
next_seven_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&units=metric&exclude=hourly,minutely,alerts&appid={WEATHER_API_KEY}"


def get_next_seven_days(next_seven_url):
    resp = requests.get(url=next_seven_url).json()
    weather_list = []
    for day in resp['daily']:
        unix_date = int(day['dt'])
        weather_date = datetime.utcfromtimestamp(
            unix_date).strftime('%Y-%m-%d')
        weather_temp = day['temp']['day']
        weather_main = day['weather'][0]['main']

        weather_list.append([weather_date, weather_temp, weather_main])

    df_weather = pd.DataFrame(weather_list, columns=[
                              'date', 'average_temp', 'type'])
    print(df_weather)
    return df_weather


df_weather = get_next_seven_days(next_seven_url)
