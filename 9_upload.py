from utils import *
import os


# 9월
names = ['김유신', '박재훈', '백보현', '신경호',
         '윤희우', '이길무', '이현국', '조경미', '홍무궁']
# 10월
# names = ['김유신', '박재훈', '백보현', '신경호', 'test bed',
#          '윤희우', '이길무', '이현국', '조경미', '홍무궁']

cols = cols_dic['ah_use_log_byminute']

for name in names:
    gateway_id = get_gateway_id(name)
    device_list = get_device_list(gateway_id)
    print(name)
    arr = os.listdir("./sample_data/csv/db_data_09/"+name)
    device_names = [x[4:-4] for x in arr]
    for device_name in device_names:
        df = pd.read_csv(f'./sample_data/csv/db_data_09/{name}/{name}_{device_name}.csv')
        device_id = device_list.loc[device_list.device_name ==device_name, :].device_id.values[0]
        df.loc[:, 'gateway_id'] = gateway_id
        df.loc[:, 'device_id'] = device_id
        df.loc[:, 'time'] = pd.to_datetime(df.date_time, format='%Y-%m-%d %H:%M')
        df = df.set_index('time', drop=True)
        df = unpacking_time(df)
        df.loc[:, 'quality'] = 100
        df.loc[df.status.isna(), 'status'] = 0
        df.loc[:, 'appliance_status'] = df.status
        df.loc[df.data.isna(), 'data'] = 0
        df.loc[:, 'energy_diff'] = df.data
        df.loc[:, 'collect_time'] = [str(x)[11:13] + str(x)[14:16] for x in df.collect_time]
        df = df.loc[:,  cols[:-1]]
        # write_db(df, table_name='AH_USE_LOG_BYMINUTE_LABELED_sbj')
        print(device_name,':', device_id)