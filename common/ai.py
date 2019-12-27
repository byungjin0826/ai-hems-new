import pandas as pd
import settings
import common.data_load as dl


def get_one_day_schedule(device_id='000D6F000F74413A1', gateway_id='ep18270236', dayofweek=1,
                         conn=settings.conn, collect_date=None):
    date_condition = "        AND COLLECT_DATE >=  DATE_FORMAT(DATE_ADD(NOW(), INTERVAL -28 DAY), '%Y%m%d')) t"
    if collect_date is not None:
        date_condition = f"""        AND COLLECT_DATE >=  DATE_FORMAT(DATE_ADD('{collect_date}', INTERVAL -28 DAY), '%Y%m%d')
        AND COLLECT_DATE < '{collect_date}') t"""

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
{date_condition}
WHERE 1=1
AND DOW = {dayofweek+1}
GROUP BY
    DOW
    , COLLECT_TIME"""

    df = pd.read_sql(sql, con=settings.conn)

    # print(sql)

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

        # 시간 짧은 경우 제거하는 코드
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
                               'END': ['23:59:00'],
                               'STATUS': appliance_status})
    # print(sql)
    return df


def get_ai_schedule(device_id='000D6F000F74413A1', gateway_id='ep18270236'):
    df = pd.DataFrame(columns=['DAYOFWEEK', 'START', 'END', 'DURATION', 'STATUS'])
    for i in range(7):
        temp = get_one_day_schedule(device_id=device_id, gateway_id=gateway_id, dayofweek=i)
        temp['DAYOFWEEK'] = i
        df = pd.concat([df, temp], ignore_index=True)
    return df.loc[:, ['DAYOFWEEK', 'START', 'END', 'DURATION', 'STATUS']]


def get_total_energy_one_device(device_id='000D6F000F74413A1'):

    return 0


def device_energy_info(gateway_id='ep18270236', start_date=None):
    date_condition = ""
    if start_date is not None:
        date_condition = f"AND COLLECT_DATE >= {start_date}"

    sql = f"""
SELECT 
    DEVICE_ID
    , CASE WHEN APPLIANCE_STATUS = 1 THEN 'ON' ELSE 'OFF' END APPLIANCE_STATUS
    , avg(ENERGY_DIFF) ENERGY_DIFF
FROM AH_USE_LOG_BYMINUTE
WHERE 1=1
AND GATEWAY_ID = '{gateway_id}'
AND APPLIANCE_STATUS is not NULL
{date_condition}
GROUP BY
    DEVICE_ID
    , APPLIANCE_STATUS
"""

    df = pd.read_sql(sql, con=settings.conn)
    df_pivot = df.pivot_table(index='DEVICE_ID', columns='APPLIANCE_STATUS', values='ENERGY_DIFF')
    df_pivot.loc[df_pivot.OFF.isna(), 'OFF'] = df_pivot.loc[df_pivot.OFF.isna(), 'ON']
    device_info = dl.device_info(gateway_id=gateway_id)

    result = device_info.merge(df_pivot, on='DEVICE_ID')
    return result


def ai_simulation(gateway_id='ep18270236'):
    df = device_energy_info(gateway_id=gateway_id, start_date='20191101')
    df['SAVING'] = 0

    for device_id in df.DEVICE_ID:

        df.loc[df.DEVICE_ID == device_id, 'SAVING'] = sum([float(x) for x in get_ai_schedule(device_id=device_id).DURATION])
    return df


def merge_log_and_schedule(device_id='000D6F0012577B441', collect_date='20191101'):
    sql = f"""
SELECT 
    COLLECT_TIME
    , APPLIANCE_STATUS
FROM AH_USE_LOG_BYMINUTE
WHERE 1=1
AND DEVICE_ID = '{device_id}'
AND COLLECT_DATE = '{collect_date}'"""

    df = pd.read_sql(sql, con=settings.conn)

    ai_schedule = get_one_day_schedule(device_id=device_id, collect_date=collect_date)
    # print(ai_schedule)
    ai_schedule['START'] = [x[:2] + x[3:5] for x in ai_schedule['START']]
    ai_schedule['END'] = [x[:2] + x[3:5] for x in ai_schedule['END']]

    ai_schedule_transform = pd.DataFrame({'COLLECT_TIME':[str(x//60).rjust(2, '0') + str(x%60).rjust(2, '0') for x in range(1440)],
                                          'APPLIANCE_STATUS':[None for x in range(1440)]})

    for i in ai_schedule.iterrows():
        START = i[1][0]
        END = i[1][1]
        STATUS = i[1][3]
        ai_schedule_transform.loc[(ai_schedule_transform.COLLECT_TIME >= START) & (ai_schedule_transform.COLLECT_TIME <= END), 'APPLIANCE_STATUS'] = STATUS

    df['SCHEDULE'] = ai_schedule_transform.APPLIANCE_STATUS
    return df


def compare_schedule(collect_date='20191101'):
    return 0


if __name__ == '__main__':
    # one_day_schedule1 = get_one_day_schedule(device_id='000D6F001257DA4B1', dayofweek=1, collect_date='20191112')
    # one_day_schedule2 = get_one_day_schedule(device_id='000D6F0012577B441', dayofweek=1, collect_date='20191008')
    # schedule = get_ai_schedule(device_id='000D6F001257E2981')
    # df = device_energy_info(start_date='20191101')
    # device_list = dl.device_info(house_name='안채').DEVICE_ID
    # df = ai_simulation()
    # for device_id in device_list:
    #     device_ = sum([float(x) for x in get_ai_schedule(device_id='00158D000151B32B1').DURATION])
    #
    # print('complete')
    #
## 작업중.
    # acc_table = pd.DataFrame(columns=['device_id', 'date', 'acc', 'type1', 'type2'])
    # devices = dl.device_info().DEVICE_ID
    # dates = [20191101 + x for x in range(31)]
    # i = 1
    # for device in devices:
    #     print(f'{i}/{len(devices)}')
    #     for date in dates:
    #         df = merge_log_and_schedule(device_id=device, collect_date=date)
    #         acc = len(df.loc[df.APPLIANCE_STATUS != df.SCHEDULE, :])
    #         type1 = len(df.loc[(df.APPLIANCE_STATUS==1) & (df.SCHEDULE==0), :])  # 불편
    #         type2 = len(df.loc[(df.APPLIANCE_STATUS==0) & (df.SCHEDULE==1), :])  # 에너지 낭비
    #         # print(f"\t{date} \n\tacc: {acc}, 불편: {type1}, 에너지 낭비: {type2}")
    #         acc_table = acc_table.append({'device_id':device, 'date':date, 'acc':acc, 'type1':type1, 'type2':type2}, ignore_index=True)
    #     i+=1
    #
    # df = compare_schedule(device_id='00158D000151B49A1', collect_date='20191112')

    # data = ["first", "second", "third"]
    # for name in data:
    #     globals()[name] = [x for x in range(3)]
    #
    # print('complete')
    # gateway_id = dl.device_info(house_name = '안채').GATEWAY_ID[0]
    # energy_info = device_energy_info(gateway_id=gateway_id)

    schedule1 = get_one_day_schedule(device_id = '00158D0001A481ED1', collect_date='20191101')

