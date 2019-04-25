SELECT gateway_id, device_address, collected_date, collected_time, NAME, onoff, energy
FROM ah_log_socket_201903
WHERE 1=1
AND gateway_id = 'ep1827-dpiuctns-0236'
AND device_address = '000D6F0012577B44'