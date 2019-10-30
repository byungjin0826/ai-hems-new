request_dr_no = '2019101502'
house_no = '20180810000008'


sql = f"""
SELECT
	DEVICE_ID
	, DEVICE_NAME
	, FREQUENCY
	, WAIT_ENERGY_AVG
	, USE_ENERGY_AVG
    , FLAG_USE_AI
	, STATUS
	, ONOFF
	, ENERGY
	, USE_ENERGY_AVG * (SELECT TIMESTAMPDIFF(MINUTE, START_DATE, END_DATE) FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}') as ENERGY_SUM
FROM
(SELECT
	FR.DEVICE_ID
	, T.DEVICE_NAME
	, FR.FREQUENCY
	, (case when HT.WAIT_ENERGY_AVG is null then 0 else HT.WAIT_ENERGY_AVG end) WAIT_ENERGY_AVG
	, (case when HT.USE_ENERGY_AVG is null then 0 else HT.USE_ENERGY_AVG end) USE_ENERGY_AVG
	, T.STATUS
	, T.ONOFF
	, case 
	when (case when T.STATUS = 1 then HT.USE_ENERGY_AVG else HT.WAIT_ENERGY_AVG end) is null then 0
	else (case when T.STATUS = 1 then HT.USE_ENERGY_AVG else HT.WAIT_ENERGY_AVG end) end as ENERGY
    , case when FLAG_USE_AI = 'Y' then 1 else 0 end FLAG_USE_AI
FROM (
	SELECT 
		DEVICE_ID
		, sum(APPLIANCE_STATUS)
		, case when sum(APPLIANCE_STATUS) is null then 0 else sum(APPLIANCE_STATUS) end FREQUENCY
	FROM
		(SELECT 
			COLLECT_DATE
			, DEVICE_ID
			, max(APPLIANCE_STATUS) APPLIANCE_STATUS
		FROM AH_USE_LOG_BYMINUTE
		WHERE 1=1
		AND GATEWAY_ID = (
			SELECT GATEWAY_ID
			FROM AH_GATEWAY_INSTALL
			WHERE 1=1
			AND HOUSE_NO = '{house_no}'
		)
		AND DAYOFWEEK(COLLECT_DATE) = (SELECT DAYOFWEEK(DATE_FORMAT(START_DATE, '%Y%m%d')) FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}')
		AND COLLECT_TIME >= (SELECT DATE_FORMAT(START_DATE, '%H%i') FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}')
		AND COLLECT_TIME <= (SELECT DATE_FORMAT(END_DATE, '%H%i') FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}')
		GROUP BY
			COLLECT_DATE
			, DEVICE_ID) t
		GROUP BY
			DEVICE_ID
		) FR
	INNER JOIN 
		(SELECT
			DEVICE_ID
			, sum(WAIT_ENERGY)/sum(WAIT_TIME) WAIT_ENERGY_AVG
			, sum(USE_ENERGY)/sum(USE_TIME) USE_ENERGY_AVG
		FROM AH_DEVICE_ENERGY_HISTORY
		WHERE 1=1
		AND GATEWAY_ID = (
			SELECT GATEWAY_ID
			FROM AH_GATEWAY_INSTALL
			WHERE 1=1
			AND HOUSE_NO = '{house_no}')
		GROUP BY
			DEVICE_ID) HT
	ON FR.DEVICE_ID = HT.DEVICE_ID
	INNER JOIN 
		(SELECT 
			gateway_id
			, device_id
			, device_name
		    , sum(onoff) onoff_sum
		    , count(onoff) onoff_count
		    , avg(POWER) power_avg
		    , case when sum(onoff) > 2.5 then 1 else 0 end onoff
			, case when avg(POWER) > 0.5 then 1 else 0 end status -- 조정필요
		FROM aihems_service_db.AH_LOG_SOCKET
		WHERE 1=1
		AND GATEWAY_ID = (SELECT GATEWAY_ID FROM aihems_api_db.AH_GATEWAY_INSTALL WHERE 1=1 AND HOUSE_NO = '{house_no}')
		AND COLLECT_DATE = (SELECT DATE_FORMAT(NOW(), '%Y%m%d') FROM DUAL)
		-- DATE_FORMAT(DATE_ADD((SELECT START_DATE FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}'), INTERVAL -5 DAY), '%Y%m%d')
		AND COLLECT_TIME >= DATE_FORMAT(DATE_ADD(DATE_ADD(NOW(), INTERVAL 9 HOUR), INTERVAL -5 MINUTE), '%H%i')
		GROUP BY
			gateway_id
			, device_id
			, device_name) T on (FR.DEVICE_ID = T.DEVICE_ID)
    INNER JOIN
		(SELECT
			DEVICE_ID
			, FLAG_USE_AI
		FROM AH_DEVICE) QQ ON FR.DEVICE_ID = QQ.DEVICE_ID
	) FF
ORDER BY
    FLAG_USE_AI asc
	, STATUS desc
	, ONOFF desc
	, FREQUENCY desc
	, ENERGY asc
            """


print(sql)