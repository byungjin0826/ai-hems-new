from utils import *

gateway_id = get_gateway_id('박재훈')

device_list = get_device_list(gateway_id)

schedule = calc_weekly_schedule(device_list.device_id[4])
# print(device_list)
#
# device_ids = device_list.loc[device_list.device_type == 'socket', 'device_id']
#
# for device_id in device_ids:
#     df = calc_appliance_energy_history(device_id)
#     write_db(df, table_name='AH_APPLIANCE_ENERGY_HISTORY')
#
# # gateway_id = 'ep18270236'
# #
# df = get_usage_hourly(gateway_id)
#
# write_db(df, table_name='ah_usage_hourly')
#