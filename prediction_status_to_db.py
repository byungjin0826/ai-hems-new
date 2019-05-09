from utils import *

sql = """
SELECT REPLACE(CURRENT_DATE()-INTERVAL 1 DAY,'-','') AS today
"""
today = get_table_from_db(sql, db='aihems_api_db')
year_date = today.today[0]