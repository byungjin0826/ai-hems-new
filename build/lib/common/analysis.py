"""
analysis
"""

import pandas as pd
import numpy as np
import common.data_load as dl
import settings


class House: # todo: 나중에 작업.
    def __init__(self):
        self.house_name = None
        self.device_list = None
        self.start_date = None
        self.end_date = None
        self.status = None
        self.device_info = None


def schedule_check(device_name, house_name, dayofweek):
    start_date = '20191101'

    device_id = dl.device_info(device_name=device_name, house_name=house_name).DEVICE_ID[0]
    log = dl.usage_log(device_id=device_id, start_date=start_date, dayofweek=dayofweek)


def device_energy_per_hour(gateway_id='ep18270236', start_date=None, end_date=None):
    sql = f"""
SELECT
	DEVICE_ID
	, APPLIANCE_STATUS
	, AVG(ENERGY_DIFF)
FROM AH_USE_LOG_BYMINUTE
WHERE 1=1
AND GATEWAY_ID = '{gateway_id}'
GROUP BY
	DEVICE_ID
	, APPLIANCE_STATUS
"""
    df = pd.read_sql(sql, con=settings.conn)

    return df


if __name__ == '__main__':
    # schedule_check(house_name='윤희우', device_name='건조기', dayofweek=5)
    a = House()

