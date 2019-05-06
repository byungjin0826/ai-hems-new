from utils import *

gateway_id = get_gateway_id('안채')

device_list = get_device_list(gateway_id)

# print(device_list)

device_id = '000D6F001257E2981'

df = get_appliance_energy_history(device_id)

write_db(df, table_name='AH_APPLIANCE_ENERGY_HISTORY')

