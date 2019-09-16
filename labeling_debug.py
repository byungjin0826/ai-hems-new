from utils import *



if __name__ == '__main__':
    gateway_id = 'ep17470130'
    device_id = '00158D0001524B081'
    collect_date =  '20190907'

    start = collect_date + '0000'
    end = collect_date + '2359'

    sql = f"""
SELECT    *
FROM      AH_USE_LOG_BYMINUTE
WHERE      1=1
   AND   GATEWAY_ID = '{gateway_id}'
   AND   DEVICE_ID = '{device_id}'
   AND   CONCAT( COLLECT_DATE, COLLECT_TIME) >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{start}', '%Y%m%d%H%i'),INTERVAL -20 MINUTE), '%Y%m%d%H%i')
     AND   CONCAT( COLLECT_DATE, COLLECT_TIME) <= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{end}', '%Y%m%d%H%i'),INTERVAL 10 MINUTE), '%Y%m%d%H%i')
ORDER BY COLLECT_DATE, COLLECT_TIME
"""
    df = get_table_from_db(sql)

    print(df.head())
    print('df:', len(df))

    x, y = split_x_y(df, x_col='energy_diff')

    pre = 20
    post = 10
    length = post + pre

    x = [x[i:i + length] for i in range(len(x) - (pre + post))]

    # model = load(f'./sample_data/joblib/by_device/{device_id}_labeling.joblib')

    model = load(f'./sample_data/joblib/by_device/000D6F000F745CBD1_labeling.joblib')

    y = model.predict(x)

    y = [int(x) for x in y]