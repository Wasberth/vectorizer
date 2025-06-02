import mysql.connector
from dotenv import load_dotenv
import os
load_dotenv()

database = mysql.connector.connect(host=os.environ['db_host'], user=os.environ['db_user'], password=os.environ['db_pass'], database=os.environ['db_name'])