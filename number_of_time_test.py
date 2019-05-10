from utils import *

def auto_dr():
    cbl = 0
    saving_money = 0
    recommendation_list = 0
    return cbl, saving_money, recommendation_list

def calc_reccomendation_table(gate_way_id, max_usable_energy):
    start = '08:00'
    end = '09:00'
    result = pd.DataFrame(columns=['device_id', 'frequency', 'max_energy', 'min_duration'])
    device_list_socket = ['000D6F001257586D1', # gateway_id를 이용해서 만들어야
                          '000D6F00125772121',
                          '000D6F001257791D1',
                          '000D6F0012577B441',
                          '000D6F001257D76F1',
                          '000D6F001257D8BA1']

    for device_id in device_list_socket:
        print(device_id)
        sql = f"""
        SELECT *
        FROM AH_USE_LOG_BYMINUTE_LABELED_copy
        WHERE device_id = '{device_id}'
        """
        df = get_table_from_db(sql)
        summary=get_usage_summary(df)
        temp = {'device_id':device_id,
        'frequency' : calc_number_of_time_use(device_id=device_id, start=start, end = end),
        'max_energy' : max(summary.sum_of_energy_diff),
        'min_duration' : min(summary.duration)}
        result = result.append(temp, ignore_index=True)

    result = result.sort_values('max_energy').reset_index(drop=True)
    result.loc[:, 'cumsum_energy'] = result.max_energy.cumsum()
    result.loc[:, 'usable'] = result.cumsum_energy < max_usable_energy
    return result

max_usable_energy = calc_cbl('ep18270236', date = '2019-03-31', start='08:00', end = '09:00')

df = calc_reccomendation_table(400, max_usable_energy)