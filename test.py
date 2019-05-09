from utils import *

sql = f"""
SELECT *
FROM AH_USE_LOG_BYMINUTE_LABELED_copy
WHERE 1=1
AND GATEWAY_ID = 'ep17470141'
AND DEVICE_ID = '00158D000151B1F91'
"""

df = get_table_from_db(sql)


def calc_number_of_times(device_id, start = '00:00', end = '00:45'):
    date = datetime.datetime.now().strftime('%Y%m%d')
    df = get_raw_data(device_id=device_id)   # table 향후 변경 필요
    df = binding_time(df)[:date]

    start_time = datetime.time(int(start[:2]), int(start[-2:]))
    end_time = datetime.time(int(end[:2]), int(end[-2:]))

    df_hourly_per_15min = df.loc[:, 'appliance_status'].resample('15min').max()

    df_hourly_per_15min_subset = df_hourly_per_15min[start_time:end_time]

    cbl_list = [x for x in df_hourly_per_15min_subset.resample('1d').sum()[-6:-1]]
    cbl_list.sort()
    print(cbl_list)
    return 0