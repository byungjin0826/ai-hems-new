import pymysql
import settings
import common.ai as ai
import common.data_load as dl
import pandas as pd


if __name__ == '__main__':
    device_id = '00158D0001524B0F1'
    # updat
    dl.label_modify(device_id=device_id, collect_date_range=('20191101', '20191209'), threshold=5)
    df = dl.usage_log(device_id=device_id, start_date='20191107', raw_data=True)

    # df = ai.merge_log_and_schedule(device_id = device_id, collect_date = '20191101')
    log = dl.usage_log(device_id = device_id, start_date='20191101')
    sql = f"""
SELECT 
    POWER
    , COUNT(*) CNT
FROM
    (SELECT 
        STR_TO_DATE(CONCAT(COLLECT_DATE, COLLECT_TIME), '%Y%m%d%H%i') DATETIME
        , COLLECT_DATE
        , COLLECT_TIME
        , POWER
        , ENERGY_DIFF
        , ONOFF
        , APPLIANCE_STATUS
    FROM AH_USE_LOG_BYMINUTE
    WHERE 1=1
    AND DEVICE_ID = '{device_id}'
    AND COLLECT_DATE >= '20191101'
    AND COLLECT_TIME >= '0000'AND COLLECT_TIME <= '2359')
    t
GROUP BY
    POWER
    """

    power_table = pd.read_sql(sql, con=settings.conn)
