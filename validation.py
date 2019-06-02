import utils

# A: device_id, accuracy
# B: appliance_no, gateway_id, device_id, appliance_type, appliance_name

a =


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



