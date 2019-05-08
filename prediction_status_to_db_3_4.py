from utils import *

# # 일단 모델을 로컬에 저장
# member_name = input('member name : ')
# appliance_name = input('appliance name : ')
# # device_address = input('device_address:')
#
# device_address = search_device_address(member_name, appliance_name)

query = """
SELECT distinct(device_id)
FROM AH_USE_LOG_BYMINUTE_LABELED
"""
device_address_list = get_table_from_db(query, db='aihems_api_db')

for device_address in device_address_list.device_id:
    if device_address == '00158D0001A42F8A1':
        continue
    # device_address = device_address[:-1]
    print(device_address)
    gs = load('./sample_data/joblib/'+device_address+'_labeling.joblib')

    lag = 10

    sql = f"""
    SELECT AL.*
    FROM ah_device AS AD
    JOIN
    ah_log_socket_201903 AS AL
    ON AD.gateway_id = AL.gateway_id
    WHERE 1=1
    AND AL.device_address = '{device_address}'
    UNION
    SELECT AL.*
    FROM ah_device AS AD
    JOIN
    ah_log_socket_201904 AS AL
    ON AD.gateway_id = AL.gateway_id
    WHERE 1=1
    AND AL.device_address = '{device_address}'
    """

    # gateway_id = ""
    # device_address_condition = f"AND LOG3.device_address = '{device_address}'"
    # device_address_condition_1 = f"AND LOG4.device_address = '{device_address}'"
    # gateway_id_condition = f"AND gateway_id = {gateway_id}"

    # sql += device_address_condition
    # sql += gateway_id_conditon

    df = labeling_db_to_db(sql, db='aihems_service_db')

    x, y = split_x_y(df, x_col='energy_diff', y_col='appliance_status')

    x, y = sliding_window_transform(x,y,lag=lag,step_size=30)

    df = df.iloc[:-lag]

    df['appliance_status'] = gs.predict(x)

    write_db(df)



