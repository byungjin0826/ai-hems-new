from utils import *

tv_type = 'F0'

sql = """
SELECT DISTINCT(DEVICE_ID)
FROM AH_APPLIANCE
LEFT JOIN AH_APPLIANCE_HISTORY
ON AH_APPLIANCE.APPLIANCE_TYPE = AH_APPLIANCE_HISTORY.APPLIANCE_TYPE
WHERE 1=1
AND AH_APPLIANCE.APPLIANCE_TYPE = 'F0'
"""
device_id_list_tv = [x for x in get_table_from_db(sql).device_id]

device_id_list_tv