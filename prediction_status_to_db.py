from utils import *

sql = """
SELECT REPLACE(CURRENT_DATE()-INTERVAL 1 DAY,'-','') AS today
"""
today = get_table_from_db(sql, db='aihems_api_db')
year_date = today.today[0]

#todo : byminute 테이블에 들어있는 device_id여야함
#todo : model이 있을경우와 없을경우를 둘다 고려해야함
#todo :