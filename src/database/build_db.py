import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from dotenv import load_dotenv
import requests
import os
import pymysql
import sys
import json



load_dotenv()

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")


def connect_to_mysql():
    connectionData = f"mysql+pymysql://{db_user}:{db_password}@localhost/cityplay"
    try:

        engine = create_engine(connectionData, echo=False)
        engine.execution_options(isolation_level="AUTOCOMMIT")
        if not database_exists(engine.url):
            create_database(engine.url)

        print(database_exists(engine.url))
        print("Great..we have connected to the DB")
        conn = engine.connect()

        return conn
    except Exception as error:
        print("Oh no..could not connect to the DB. Exiting...")
        print(error)
        sys.exit()


def create_schemas(conn):
    # drop all tables and re create
    # Sales Schema
    try:
        print("Creating Sales tables...")
        # conn.execute('DROP TABLE IF EXISTS Chats;')
        # conn.execute('DROP TABLE IF EXISTS Sales;')

        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS Sales(
            id int not null auto_increment primary key,
            date DATE not null,
            total_sales float not null,
            day_of_week varchar(15) not null,
            month_name varchar(15) not null,
            day int not null,
            year int not null,
            average_temp float not null,
            total_precip_mm float not null,
            did_rain int not null,
            day_type varchar(15) not null,
            holiday_type varchar(100),
            holiday_name varchar(50),
            is_closed int not null,
            is_lockdown int not null,
            is_curfew int not null,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)
            ENGINE=INNODB;
            '''
        )
        print("Finished creating sales table!")

    except Exception as error:
        print("There was an error creating the Sales Table...exiting")
        print(error)
        sys.exit()

    try:
        print("Creating Holidays table...")

        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS Holidays(
            id int not null auto_increment primary key,
            date DATE not null,
            day_type varchar(15) not null,
            holiday_type varchar(100),
            holiday_name varchar(100))
            ENGINE=INNODB;
            '''
        )
        print("Finished creating Holidays table!")

    except Exception as error:
        print("There was an error creating the Holidays Table...exiting")
        print(error)
        sys.exit()

    is_db_empty(conn)


def is_db_empty(conn):
    try:
        sales_result = conn.execute(
            '''
            select count(*) from Sales;
            '''
        )
        sales_rows = sales_result.fetchone()[0]

        if sales_rows == 0:
            insert_sales_data(conn)
        else:
            return

        holidays_result = conn.execute(
            '''
            select count(*) from Holidays;
            '''
        )

        holiday_rows = holidays_result.fetchone()[0]
        if holiday_rows == 0:
            insert_holiday_data(conn)
        else:
            return

    except Exception as error:
        print("There was an error INSERTING the INFO...exiting")
        print(error)
        sys.exit()


def insert_sales_data(conn):
    df_sales = pd.read_csv('./data/db_load_files/clean_data.csv')
    df_sales = df_sales.where(pd.notnull(df_sales), None)
    del df_sales['did_snow']
    cols = "`,`".join([str(i) for i in df_sales.columns.tolist()])

    # Insert DataFrame recrds one by one.
    for _i, row in df_sales.iterrows():
        sql = "INSERT INTO `Sales` (`" + cols + \
            "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        conn.execute(sql, tuple(row))


def insert_holiday_data(conn):
    df_holidays = pd.read_excel('./data/db_load_files/calendario.xls')
    df_holidays.rename(columns={'Dia': 'date', 'laborable / festivo / domingo festivo': 'day_type',
                       'Festividad': 'holiday_name', 'Tipo de Festivo': 'holiday_type'}, inplace=True)

    del df_holidays['Dia_semana']

    df_holidays = df_holidays.where(pd.notnull(df_holidays), None)

    cols = "`,`".join([str(i) for i in df_holidays.columns.tolist()])

    # Insert DataFrame recrds one by one.
    for _i, row in df_holidays.iterrows():
        sql = "INSERT INTO `Holidays` (`" + cols + \
            "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        conn.execute(sql, tuple(row))
