from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib
import sys
from dotenv import load_dotenv
import os
import pandas

load_dotenv()

myGMPasswd = os.getenv("myGMPasswd")


def send_email(df):

    recipients = ['fran.leston@outlook.com']
    emaillist = [elem.strip().split(',') for elem in recipients]
    msg = MIMEMultipart()
    msg['Subject'] = "CityPlay Weekly Forecast"
    msg['From'] = 'lestonramos@gmail.com'

    html = """\
    <html>
    <head></head>
    <body>
    <h1>CityPlay</h1>
    <h2>Sales Prediction for the following week:</h2>
        {0}
    </body>
    </html>
    """.format(df.to_html())

    part1 = MIMEText(html, 'html')
    msg.attach(part1)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    myGMPasswd = "uejhffcmlmrvaelm"
    server.ehlo()
    server.starttls()
    server.login(msg['From'], myGMPasswd)
    server.sendmail(msg['From'], emaillist, msg.as_string())
    server.quit()
