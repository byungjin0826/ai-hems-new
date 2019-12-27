import pymysql
import settings
import common.ai as ai
import common.data_load as dl
import pandas as pd


if __name__ == '__main__':

    # threshold 확인 작업.
    device_id = '00158D000151B4621'
    # update
    # dl.label_modify(device_id=device_id, collect_date_range=('20191101', '20191216'), threshold=1)
    df = dl.usage_log(device_id=device_id, start_date='20191119', raw_data=True)

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


#     # 평균이랑 표준편차 산정.
#     sql = f"""
# SELECT
#     DEVICE_ID
#     , AVG(CASE WHEN POWER > 0 THEN POWER ELSE NULL END) POWER_AVG
#     , STDDEV_SAMP(CASE WHEN POWER > 0 THEN POWER ELSE NULL END) POWER_STD
# FROM
#     (SELECT *
#     FROM AH_USE_LOG_BYMINUTE
#     WHERE 1=1
#     AND COLLECT_DATE >= '20191201') T01
# GROUP BY
# DEVICE_ID"""
#
#     sql = f"""
# SELECT
# 	DEVICE_ID
# 	, POWER_AVG
# 	, POWER_STD
# 	, CASE WHEN POWER_AVG - POWER_STD > 0 THEN POWER_AVG - POWER_STD ELSE 1 END THRESHOLD
# FROM
# (SELECT
# 	DEVICE_ID
# 	, AVG(POWER) POWER_AVG
# 	, STDDEV_POP(POWER) POWER_STD
# FROM
# 	(SELECT
# 		DEVICE_ID
# 		, POWER
# 	FROM
# 		(SELECT
# 			DEVICE_ID
# 			, CASE WHEN POWER < 0 THEN NULL ELSE POWER END POWER
# 		FROM
# 			AH_USE_LOG_BYMINUTE
# 		WHERE 1=1
# 		AND COLLECT_DATE >= '20191201') T01
# 	GROUP BY
# 		DEVICE_ID
# 		, POWER) T02
# GROUP BY
# 	DEVICE_ID) T03"""
#
#     threshold_info = pd.read_sql(sql, con=settings.conn)
#
#     device_info = dl.device_info()
#
#     merged = device_info.merge(threshold_info, on = 'DEVICE_ID')


