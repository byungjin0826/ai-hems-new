"""
db connection info, etc..
"""

import pymysql
from contextlib import contextmanager


host = 'aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com'
db = 'aihems_api_db'
port = 3306
user = 'aihems'
password = '#cslee1234'
charset = 'utf8'


@contextmanager
def open_db_connection():
    conn = pymysql.connect(host=host, port=port, user=user,
                           passwd=password, db=db, charset=charset)
    try:
        yield conn
    except Exception as e:
        print(e)
    finally:
        conn.close()

if __name__ == '__main__':
    with open_db_connection() as conn:
        conn.db = 'aihems_service_db'
        print(conn)