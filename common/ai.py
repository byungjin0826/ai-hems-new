import pandas as pd
import settings


def get_one_day_schedule(device_id='000D6F000F74413A1', gateway_id='ep18270236', dayofweek=1,
                         conn=settings.conn):
    sql = f"""
    SELECT
        DOW
        , COLLECT_TIME
        , STR_TO_DATE(COLLECT_TIME, '%H%i'), DATETIME
    -- 	, MAX(APPLIANCE_STATUS)
        , case when AVG(APPLIANCE_STATUS) > 0 then 1 else 0 end APPLIANCE_STATUS
    FROM (	SELECT
                STR_TO_DATE(CONCAT(COLLECT_DATE, COLLECT_TIME), '%Y%m%d%H%i') DATETIME
                , DAYOFWEEK(COLLECT_DATE) DOW
            -- 	, COLLECT_DATE
                , COLLECT_TIME
            --     , POWER
                , ENERGY_DIFF
            --     , ONOFF
                , CASE WHEN POWER >= 5 THEN 1 ELSE 0 END APPLIANCE_STATUS
            FROM AH_USE_LOG_BYMINUTE
            WHERE 1=1
            -- AND GATEWAY_ID = '{gateway_id}'
            AND DEVICE_ID = '{device_id}'
            AND COLLECT_DATE >=  DATE_FORMAT(DATE_ADD(NOW(), INTERVAL -28 DAY), '%Y%m%d')) t
    WHERE 1=1
    AND DOW = {dayofweek+1}
    GROUP BY
        DOW
        , COLLECT_TIME
    """

    df = pd.read_sql(sql, con=settings.conn)

    if sum(df.APPLIANCE_STATUS) == 0:
        df = pd.DataFrame({'START': '00:00:00', 'END': '23:59:00',
                           'DURATION': '1359', 'STATUS': 0}, index=[0])

    else:
        df['APPLIANCE_STATUS_LAG'] = df.APPLIANCE_STATUS.shift(1).fillna(0)
        df['APPLIANCE_STATUS_LAG'] = [int(x) for x in df.APPLIANCE_STATUS_LAG]
        df['APPLIANCE_STATUS'] = df.APPLIANCE_STATUS.fillna(0)
        df['APPLIANCE_STATUS'] = [int(x) for x in df.APPLIANCE_STATUS]

        df_change = df.loc[df.APPLIANCE_STATUS != df.APPLIANCE_STATUS_LAG, :]
        df_change = df_change.reset_index()
        df_change['DATETIME_LAG'] = df_change.DATETIME.shift(1).fillna(0)
        df_change.iloc[0, 7] = pd.to_datetime(df_change.iloc[0, 4].strftime('%Y%m%d' + ' 00:00'))
        df_change['duration'] = df_change.DATETIME - df_change.DATETIME.shift(1)
        df_change['duration'] = [x.seconds/60 for x in df_change.duration if x != 'NaT']

        history = df_change.loc[:, ['DATETIME_LAG', 'DATETIME', 'duration', 'APPLIANCE_STATUS_LAG']]

        history.columns = ['START', 'END', 'DURATION', 'STATUS']
        history = history.loc[:, :].reset_index(drop=True)
        history.iloc[0, 2] = (history.iloc[0, 1]-history.iloc[0, 0]).seconds/60

        df = history

        df = df.loc[(df.index == 0) | ((df.DURATION >= 60) & (df.STATUS == 0)) |
                    ((df.DURATION >= 30) & (df.STATUS == 1)), :].reset_index(drop=True)

        status_temp = None
        for one_row in df.iterrows():
            if status_temp == one_row[1]['STATUS']:
                df = df.drop(index=one_row[0])
            status_temp = one_row[1]['STATUS']
        start_temp = df.END.iloc[-1] + pd.Timedelta('1 minutes')
        if status_temp == 1:
            status_temp = 0
        else:
            status_temp = 1
        df.START = [x.strftime('%H:%M:%S') for x in df.START]
        df.END = [x.strftime('%H:%M:%S') for x in df.END]

        df = pd.concat([df, pd.DataFrame({'START': start_temp.strftime('%H:%M:%S'),
                                          'END': '23:59:00',
                                          'DURATION': 0.0,
                                          'STATUS': status_temp},
                                         index=[one_row[0] + 1])], ignore_index=True)

        sql = f"""
SELECT ALWAYS_ON
FROM AH_DEVICE_MODEL
WHERE 1=1
AND DEVICE_ID = '{device_id}'"""

        always_on = pd.read_sql(sql, con=conn).iloc[0][0]

        if always_on == 1:
            dayofweek = [str(dayofweek)]
            time = ['00:00:00']
            appliance_status = ['1']

            df = pd.DataFrame({'DAYOFWEEK': dayofweek,
                               'START': time,
                               'STATUS': appliance_status})
    return df


def get_ai_schedule(device_id='000D6F000F74413A1', gateway_id='ep18270236'):
    df = pd.DataFrame(columns=['DAYOFWEEK', 'START', 'END', 'DURATION', 'STATUS'])
    for i in range(7):
        temp = get_one_day_schedule(device_id=device_id, gateway_id=gateway_id, dayofweek=i)
        temp['DAYOFWEEK'] = i
        df = pd.concat([df, temp], ignore_index=True)
    return df


if __name__ == '__main__':
    one_day_schedule = get_one_day_schedule(device_id='00158D000151B32B1', dayofweek=1)
    schedule = get_ai_schedule(device_id='00158D000151B32B1')
