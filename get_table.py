import pandas as pd
import pymysql


def get_table(sql, db = 'aihems_api_db'):
    conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                                port=3306, user='aihems', passwd='#cslee1234', db=db,
                                                charset='utf8')
    df = pd.read_sql(sql, con=conn)
    conn.close()
    return(df)

if __name__ == '__main__':
    sql = """
SELECT
    t2.DEVICE_ID
    , t1.APPLIANCE_NO
    , t1.APPLIANCE_TYPE
FROM AH_APPLIANCE t1
INNER JOIN 
    (select *
    from AH_APPLIANCE_CONNECT
    where flag_delete = 'N')
    t2
ON t1.APPLIANCE_NO = t2.APPLIANCE_NO
    """
    df = get_table(sql)
    print(df.head())