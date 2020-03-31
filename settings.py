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


def db_connector(func):
    def decorated():
        conn = pymysql.connect(host=host, port=port, user=user,
                               passwd=password, db=db, charset=charset)
        func()
        conn.close()
    return decorated()


if __name__ == '__main':
    test()