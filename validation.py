import utils
import utils
import os
from joblib import load
import pandas as pd

# A: device_id, accuracy
# B: appliance_no, gateway_id, device_id, appliance_type, appliance_name

device_list = []
model_score_list = []

for root, dirs, files in os.walk('./sample_data/joblib/by_device/'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        model = load(full_fname)

        device_list.append(fname[:17])
        model_score_list.append(model.best_score_)


dic = {'device_id':device_list, 'score':model_score_list}

df = pd.DataFrame(dic)

sql_b = f"""
SELECT a.APPLIANCE_NO, a.GATEWAY_ID, a.DEVICE_ID, b.APPLIANCE_TYPE, b.APPLIANCE_NAME
FROM AH_APPLIANCE_CONNECT a
LEFT JOIN (	SELECT *
			FROM AH_APPLIANCE
			WHERE 1=1
			AND FLAG_DELETE = 'n') b
ON a.APPLIANCE_NO = b.APPLIANCE_NO
"""

b = utils.get_table_from_db(sql_b)