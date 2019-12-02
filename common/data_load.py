import pandas as pd
import settings


def usage_log(device_id, gateway_id=None, start_date='20191128', end_date=None,
              start_time='0000', end_time='2359', dayofweek=None, raw_data=False):
    """

    :param device_id:
    :param gateway_id:
    :param start_date:
    :param end_date:
    :param start_time:
    :param end_time:
    :param dayofweek:
    :param raw_data:
    :return:
    """

    def make_sql():
        sql = f"""
SELECT 
    STR_TO_DATE(CONCAT(COLLECT_DATE, COLLECT_TIME), '%Y%m%d%H%i') DATETIME
    , POWER
    , ENERGY_DIFF
    , ONOFF
    , APPLIANCE_STATUS
FROM AH_USE_LOG_BYMINUTE
WHERE 1=1
AND DEVICE_ID = '{device_id}'"""
        if gateway_id is not None:
            gateway_id_condition = f"\nAND GATEWAY_ID = '{gateway_id}'"
            sql += gateway_id_condition

        start_date_condition = f"\nAND COLLECT_DATE >= '{start_date}'"
        sql += start_date_condition

        if end_date is not None:
            end_date_condition = f"\nAND COLLECT_DATE <= '{end_date}'"
            sql += end_date_condition

        start_time_condition = f"\nAND COLLECT_TIME >= '{start_time}'"
        sql += start_time_condition

        end_time_condition = f"AND COLLECT_TIME <= '{end_time}'"
        sql += end_time_condition

        if dayofweek is not None:
            dayofweek_condition = f"AND DAYOFWEEK(COLLECT_DATE) = {dayofweek}"
            sql += dayofweek_condition
        return sql

    def transform(df=pd.read_sql(make_sql(), con=settings.conn, index_col='DATETIME')):
        df['APPLIANCE_STATUS_LAG'] = df.APPLIANCE_STATUS.shift(1).fillna(0)
        df['APPLIANCE_STATUS_LAG'] = [int(x) for x in df.APPLIANCE_STATUS_LAG]
        df['APPLIANCE_STATUS'] = df.APPLIANCE_STATUS.fillna(0)
        df['APPLIANCE_STATUS'] = [int(x) for x in df.APPLIANCE_STATUS]

        df_change = df.loc[df.APPLIANCE_STATUS != df.APPLIANCE_STATUS_LAG, :]
        df_change = df_change.reset_index()
        df_change['DATETIME_LAG'] = df_change.DATETIME.shift(1).fillna(0)
        df_change['duration'] = df_change.DATETIME - df_change.DATETIME.shift(1)
        df_change['duration'] = [x.seconds / 60 for x in df_change.duration if x != 'NaT']

        history = df_change.loc[:, ['DATETIME_LAG', 'DATETIME', 'duration', 'APPLIANCE_STATUS_LAG']]

        history.columns = ['START', 'END', 'DURATION', 'STATUS']
        history = history.loc[history.STATUS == 1, :].reset_index(drop=True)
        return history

    if raw_data:
        result = pd.read_sql(make_sql(), con=settings.conn, index_col='DATETIME')

    else:
        result = transform()

    # settings.conn.close()
    return result


def status(device_id):
    sql = f"""
SELECT *
FROM aihems_service_db.ah_log_socket
WHERE 1=1
AND device_id = '{device_id}'
AND 
"""

    # status = pd.read_sql(sql, con=settings.conn)

    return sql


def status_all_device(gateway_id):
    device_sal = f"""
{gateway_id}
"""
    return device_sal


# 편리하게 찾을 수 있는 기능 구현...
def search_info(device_name=None, gateway_id=None, gateway_name=None, house_name=None,
                house_id=None):
    def sql():
        device_info_sql = f"""
SELECT *
FROM
    (SELECT
        t01.DEVICE_ID
        , t01.DEVICE_NAME
        , t06.APPLIANCE_NAME
        , t01.DEVICE_TYPE
        , t01.FLAG_USE_AI
        , t02.GATEWAY_ID
        , t03.APPLIANCE_NO
        , t04.HOUSE_NO
        , t05.HOUSE_NAME
    FROM
        AH_DEVICE t01
    INNER JOIN
        AH_DEVICE_INSTALL t02
    ON t01.DEVICE_ID = t02.DEVICE_ID
    INNER JOIN
        AH_APPLIANCE_CONNECT t03
    ON t01.DEVICE_ID = t03.DEVICE_ID
    INNER JOIN
        AH_GATEWAY_INSTALL t04
    ON t03.GATEWAY_ID = t04.GATEWAY_ID
    INNER JOIN
        AH_HOUSE t05
    ON t04.HOUSE_NO = t05.HOUSE_NO
    INNER JOIN
        AH_APPLIANCE t06
    ON t03.APPLIANCE_NO = t06.APPLIANCE_NO
    WHERE 1=1
    AND t06.FLAG_DELETE = 'N'
    AND t01.FLAG_DELETE = 'N'
    AND t03.FLAG_DELETE = 'N') t
WHERE 1=1"""

        if device_name is not None:
            device_name_condition = f"\nAND DEVICE_NAME like '%{device_name}%'"
            device_info_sql += device_name_condition

        if gateway_id is not None:
            gateway_id_condition = f"\nAND GATEWAY_ID = '{gateway_id}'"
            device_info_sql += gateway_id_condition

        if gateway_name is not None:
            gateway_name_condition = f"\nAND "
            device_info_sql += gateway_name_condition

        if house_name is not None:
            house_name_condition = f"\nAND HOUSE_NAME like '%{house_name}%'"
            device_info_sql += house_name_condition

        if house_id is not None:
            house_id_condition = f""
            device_info_sql += house_id_condition

        return device_info_sql

    df = pd.read_sql(sql(), con=settings.conn)
    # settings.conn.close()
    return df


if __name__ == '__main__':
    device_id = search_info(device_name='세탁기', house_name='윤희우').DEVICE_ID[0]
    log = usage_log(device_id=device_id, start_date='20191101', dayofweek=1)
    raw = usage_log(device_id=device_id, start_date='20191101', dayofweek=1, raw_data=True)
