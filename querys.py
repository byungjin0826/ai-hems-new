from utils import get_table_from_db

member_name = ""
gateway_id = ""
device_address = ""
member_id = ""


ah_member = f'''
SELECT member_id, member_name, house_no
FROM ah_member
WHERE 1=1
member_name = {member_name}
'''

ah_gateway_assign = f"""
SELECT member_id, gateway_id
FROM ah_gateway_assign
WHERE 1=1
member_id = {member_id}
"""

ah_device = f"""
SELECT gateway_id, device_address, device_name, appliance_no
FROM ah_device
WHERE 1=1
AND gateway_id = {gateway_id}
AND device_type = 'socket'
"""

ah_log_socket = f"""
SELECT gateway_id, device_address, collected_date, collected_time, NAME, onoff, energy
FROM ah_log_socket_201903
WHERE 1=1
AND gateway_id = {gateway_id}
AND device_address = {device_address}
"""

get_table_from_db(ah_log_socket)