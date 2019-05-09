from utils import *

query = """
SELECT distinct(device_id), gateway_id
FROM AH_USE_LOG_BYMINUTE_LABELED_copy
"""
device_address_df = get_table_from_db(query, db='aihems_api_db')
device_address_list = device_address_df.values.tolist()
for device in device_address_list:
    if device[0] == '00158D0001A42F8A1':
        continue
    device_address = device[0]
    gateway_id = device[1]
    gs = load('./sample_data/joblib/'+device_address+'_labeling.joblib')
    test_bed_list = ['ep18270486', 'ep18270236', 'ep18270363']
    lag = 10
    if gateway_id in test_bed_list:
        sql = f"""
        SELECT AUL.*
        FROM AH_GATEWAY AS AG
        JOIN AH_USE_LOG_BYMINUTE_201904 AS AUL
        ON AG.GATEWAY_ID=AUL.GATEWAY_ID
        WHERE AUL.DEVICE_ID='{device_address}'
        """
    else:
        sql = f"""
        SELECT AUL.*
        FROM AH_GATEWAY AS AG
        JOIN AH_USE_LOG_BYMINUTE_201903 AS AUL
        ON AG.GATEWAY_ID=AUL.GATEWAY_ID
        WHERE AUL.DEVICE_ID='{device_address}'
        UNION
        SELECT AUL.*
        FROM AH_GATEWAY AS AG
        JOIN AH_USE_LOG_BYMINUTE_201904 AS AUL
        ON AG.GATEWAY_ID=AUL.GATEWAY_ID
        WHERE AUL.DEVICE_ID='{device_address}'
        """

    df = labeling_db_to_db(sql, db='aihems_api_db')
    if df.empty:
        print('empty : '+device_address)
        continue
    x, y = split_x_y(df, x_col='energy_diff', y_col='appliance_status')

    x, y = sliding_window_transform(x,y,lag=lag,step_size=30)

    df = df.iloc[:-lag]

    df['appliance_status'] = gs.predict(x)
    df_dic = df.iloc[0].to_dict()
    key = df_dic['gateway_id']+'-'+df_dic['device_id']+'-'+df_dic['collect_date']+'-'+df_dic['collect_time']
    if gateway_id in test_bed_list:
        sql = f"""
            SELECT CONCAT(AUL.GATEWAY_ID,"-",AUL.DEVICE_ID,"-",AUL.COLLECT_DATE,"-",AUL.COLLECT_TIME) AS PRIMARY_KEY 
            FROM AH_GATEWAY AS AG
            JOIN AH_USE_LOG_BYMINUTE_LABELED_copy AS AUL
            ON AG.GATEWAY_ID=AUL.GATEWAY_ID
            WHERE AUL.DEVICE_ID='{device_address}'
            AND COLLECT_DATE >= '20190401' 
                """
    else:
        sql = f"""
            SELECT CONCAT(AUL.GATEWAY_ID,"-",AUL.DEVICE_ID,"-",AUL.COLLECT_DATE,"-",AUL.COLLECT_TIME) AS PRIMARY_KEY 
            FROM AH_GATEWAY AS AG
            JOIN AH_USE_LOG_BYMINUTE_LABELED_copy AS AUL
            ON AG.GATEWAY_ID=AUL.GATEWAY_ID
            WHERE AUL.DEVICE_ID='{device_address}'
            AND COLLECT_DATE >= '20190301' 
        """
    primary_key_df = get_table_from_db(sql, db='aihems_api_db')
    if primary_key_df.empty:
        primary_key = " "
    else:
        primary_key = primary_key_df.primary_key[0]
    if key == primary_key:
        print("duplicate : "+ device_address)
    else:
        write_db(df,'AH_USE_LOG_BYMINUTE_LABELED_copy')



