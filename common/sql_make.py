"""
sql
"""


def usage_log(device_id, start_date, end_date, start_time, end_time, dayofweek):
    sql = f"""
SELECT *
FROM 
WHERE 1=1
AND DEVICE_ID={device_id}
AND DATETIME>='{start_date}'
AND DATETIME<='{end_date}'
"""
    return sql


def watt_log():
    sql = f"""

"""
    return sql
