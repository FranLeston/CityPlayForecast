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


# My functions

if __name__ == '__main__':
    conn = db.connect_to_mysql()
    if conn:
        db.create_schemas(conn)
