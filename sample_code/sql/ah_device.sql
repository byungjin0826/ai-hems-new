(
	SELECT gateway_id, device_address, device_name, appliance_no
	FROM ah_device
	WHERE 1=1
	AND gateway_id = 'ep1827-dpiuctns-0236'
	AND device_type = 'socket'
)