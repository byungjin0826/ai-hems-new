import settings
from functools import wraps
import pymysql
import pandas as pd


def con_and_close(func):
    conn = pymysql.connect(host=settings.host, port=settings.port, user=settings.user,
                           passwd=settings.passwd, db=settings.db, charset=settings.charset)
    @wraps(func)
    def wrapper(*args, **kwargs):
        conn.open
        result = func(*args, **kwargs)
        conn.close()
        return result
    return wrapper


if __name__ == '__main__':
    sql = 'SELECT * FROM '
    df = read_sql(sql)
