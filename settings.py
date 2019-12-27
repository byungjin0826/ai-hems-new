"""
db connection info, etc..
"""

import pymysql


host = 'aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com'
db = 'aihems_api_db'
port = 3306
user = 'aihems'
passwd = '#cslee1234'
charset = 'utf8'

conn = pymysql.connect(host=host, port=port, user=user,
                       passwd=passwd, db=db, charset=charset)

curs = conn.cursor()
# todo: close 할 수 있도록 수정.(https://o7planning.org/en/11463/connecting-mysql-database-in-python-using-pymysql)
