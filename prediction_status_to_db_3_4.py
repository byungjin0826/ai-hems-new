from utils import *

# # 일단 모델을 로컬에 저장
# member_name = input('member name : ')
# appliance_name = input('appliance name : ')
# # device_address = input('device_address:')
#
# device_address = search_device_address(member_name, appliance_name)

query = """
SELECT distinct(device_id)
FROM AH_USE_LOG_BYMINUTE_LABELED_copy
"""
device_address_list = get_table_from_db(query, db='aihems_api_db')

for device_address in device_address_list.device_id:
    if device_address == '00158D0001A42F8A1':
        continue
    if device_address == '00158D000151B1F91':
        continue
    # if device_address == '00158D000151B2131':
    #     continue
    if device_address == '00158D000151B31C1':
        continue
    if device_address == '00158D000151B4441':
        continue
    if device_address == '00158D0001524AF71':
        continue
    if device_address == '00158D000151B4751':
        continue
    if device_address == '00158D0001A42F2F1':
        continue
    if device_address == '00158D0001A44BC51':
        continue
    if device_address == '00158D0001A459C11':
        continue
    # device_address = device_address[:-1]
    # if device_address == '000D6F001257586D1':
    #     continue
    gs = load('./sample_data/joblib/'+device_address+'_labeling.joblib')

    lag = 10

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

    # gateway_id = ""
    # device_address_condition = f"AND LOG3.device_address = '{device_address}'"
    # device_address_condition_1 = f"AND LOG4.device_address = '{device_address}'"
    # gateway_id_condition = f"AND gateway_id = {gateway_id}"

    # sql += device_address_condition
    # sql += gateway_id_conditon

    df = labeling_db_to_db(sql, db='aihems_api_db')
    if df.empty:
        print('empty : '+device_address)
        continue
    x, y = split_x_y(df, x_col='energy_diff', y_col='appliance_status')

    x, y = sliding_window_transform(x,y,lag=lag,step_size=30)

    df = df.iloc[:-lag]

    df['appliance_status'] = gs.predict(x)
    # df.reset_index()
    df_dic = df.iloc[0].to_dict()
    key = df_dic['gateway_id']+'-'+df_dic['device_id']+'-'+df_dic['collect_date']+'-'+df_dic['collect_time']
    # print(key)
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
    # print(primary_key)
    if key == primary_key:
        print("duplicate : "+ device_address)
    else:
        write_db(df,'AH_USE_LOG_BYMINUTE_LABELED_copy')



