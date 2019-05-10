from utils import *
import os

path_dir = './sample_data/joblib'

file_list = os.listdir(path_dir)

file_list.sort()

for file in file_list:
    device_id = file[:17]
    appliance_type = get_appliance_type(device_id)
    print(appliance_type, ',', device_id)

    sql = """
    SELECT DISTINCT(device_id)
    FROM AH_USE_LOG_BYMINUTE_LABELED
    """

    list = [x + '1' for x in get_table_from_db(sql).values.flatten()]
    df = get_device_list_same_type(appliance_type)
    df1 = df.loc[df.device_id.isin(list), :]
    df1 = df1.reset_index(drop=True)
    model = load_labeling_model(device_id)

    accuracys = []
    for x in df1.device_id:
        accuracy = prediction_test(model, x)

        accuracys = accuracys.append(accuracy)
        print(x, ': ', accuracy)

    print(accuracys.mean())

