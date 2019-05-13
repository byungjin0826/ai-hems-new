from utils import *
import os.path
# type 개수 확인

sql = """
SELECT A.appliance_type, COUNT(A.appliance_type) as count
FROM (SELECT p1.gateway_id, p1.DEVICE_ID, p1.appliance_type, p1.APPLIANCE_NAME
      FROM AH_APPLIANCE_HISTORY p1 LEFT JOIN AH_APPLIANCE_HISTORY p2
      ON (p1.device_id = p2.device_id AND p1.create_date < p2.create_date)
WHERE p2.create_date IS NULL) A
LEFT JOIN AH_GATEWAY_INSTALL B
ON A.gateway_id = B.gateway_id
WHERE 1=1
AND A.gateway_id NOT IN ('ep18270236', 'ep18270363', 'ep18270486')
GROUP BY A.appliance_type
"""

print(get_table_from_db(sql))

device_type = input('Appliance_type: ')



if os.path.exists(f'./sample_data/{device_type}.joblib') == False:
    model = prediction_status_model_by_type(device_type)
    dump(model, f'./sample_data/{device_type}.joblib')

test_prediction_status_by_type(device_type)