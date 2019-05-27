from utils import *
from datetime import date, timedelta
import os
#하루 전 날짜 계산

yesterday = date.today() - timedelta(1)     #하루 전 날짜 계산
yesterday = yesterday.strftime('%Y%m%d')


print(yesterday)
#하루 전 날짜 뽑아오는 쿼리
# sql = """
# SELECT REPLACE(CURRENT_DATE()-INTERVAL 1 DAY,'-','') AS today
# """
# today = get_table_from_db(sql, db='aihems_api_db')
# year_date = today.today[0]
#


Path = './sample_data/joblib/'

sql = f"""
SELECT DEVICE_ID
FROM AH_DEVICE
"""
device_id_list = get_table_from_db(sql)
cnt = 0
for device_id in device_id_list.device_id:
    file_path = Path + device_id + '_labeling.joblib'
    if os.path.isfile(file_path):            #model load / 없을경우 continue
        gs = load(file_path)
    else:
        print("no model : "+device_id)
        continue
    sql = f"""
    SELECT AUL.*
    FROM
    AH_USE_LOG_BYMINUTE AS AUL
    JOIN
    AH_GATEWAY AS AG
    ON AUL.GATEWAY_ID = AG.GATEWAY_ID
    WHERE 
    AUL.DEVICE_ID = '{device_id}'
    AND
    AUL.COLLECT_DATE = '{yesterday}'
    """
    df = get_table_from_db(sql)
    df['appliance_status'] = 0

    if df.empty:                            #데이터가 없을경우 continue
        print('empty : '+ device_id)
        continue

    lag = 10

    x, y = split_x_y(df, x_col='energy_diff', y_col='appliance_status')
    x, y = sliding_window_transform(x, y, lag=lag, step_size=30)
    df = df.iloc[:-lag]
    df['appliance_status'] = gs.predict(x)

    # write_db(df, table_name='AH_USE_LOG_BYMINUTE')      #DB UPDATE




#todo : 전체 device 목록을 불러와서 모델이 있는경우를 검사 os 모듈 사용                     #완료
#todo : 예외 : byminute 테이블에 들어있는 device_id여야함(empty처리)                        #완료
#todo : 예외 : model이 있을경우와 없을경우를 둘다 고려해야함                                #완료
#todo : 예외 : primary key는 맨 처음 뽑아온 쿼리로 조합 or data frame에서 뽑아오기