from utils import *

sql = """
SELECT DISTINCT(device_id)
FROM AH_USE_LOG_BYMINUTE_LABELED
"""
#
# x = get_table_from_db(sql)
#
# list = [x+'1' for x in get_table_from_db(sql).values.flatten()]
#
# print(get_appliance_type())
#
# type = input()
#
# df = get_device_list_same_type(type)
#
# model = load_labeling_model(df.device_id[31][:-1])
#
# df1 = df.loc[df.device_id.isin(list), :]
#
# for x in df1.device_id:
#     accuracy = prediction_test(model, x)
#     print(accuracy)
#     # print(x)

device_id = '000D6F000E4B03C61'

model = load_labeling_model(device_id[:-1])
prediction_test(model, device_id=device_id)