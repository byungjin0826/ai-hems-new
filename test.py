from utils import *


gateway_id = 'ep18270236'

# cbl = calc_cbl(gateway_id = gateway_id, date, start, end)

df = get_device_list(gateway_id) # todo: socket 만 가져오게 변경, 변경해도 문제가 안 생기는 지 확인.

df['frequency'] = df.loc[:, 'device_id'].map(lambda x: calc_number_of_time_use(x)) # todo: appliance status db 에 업데이트 필요


