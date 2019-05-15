from utils import *


# 누진 구간 예측 절차
# 일별 예측...
# 모델 학습
# 기존

# gateway_id: ''
#

gateway_id = get_gateway_id('안채')
device_list = get_device_list(gateway_id)

df = get_raw_data('')

check_date = datetime.datetime.day

today = datetime.datetime.now()

for i in range(10):
    print(i)