"""
analysis
"""

import pandas as pd
import numpy as np
import common.data_load as dl


def compare_ai():
    return 0


def schedule_check(house_name, device_name, dayofweek):

    start_date = '20191101'

    device_id = dl.search_info(device_name=device_name, house_name=house_name).DEVICE_ID[0]
    log = dl.usage_log(device_id=device_id, start_date=start_date, dayofweek=dayofweek)

    return log


if __name__ == '__main__':
    schedule_check(house_name='윤희우', device_name='건조기', dayofweek=5)
