from utils import *

df = get_raw_data(device_id = '00158D000151B1F91',table_name = 'AH_USE_LOG_BYMINUTE_LABLED')


df = binding_time(df)


schedule = df.pivot_table(values = 'appliance_status', index = df.index.time, columns = df.index.dayofweek, aggfunc='max')