select
	*
from
	(
	select
		*
	from
		AH_USE_LOG_BYMINUTE
	GROUP by
		device_id) t1
inner join AH_DEVICE_MODEL t2
	-- SELECT <열 목록>
	-- FROM <첫 번째 테이블>
	--     INNER JOIN <두 번째 테이블>
	--     ON <조인될 조건>
	-- [WHERE 검색조건]

-- sample data
 select
	DEVICE_ID
	, COLLECT_DATE
	, sum(ENERGY_DIFF) ENERGY_DIFF_SUM
from
	AH_USE_LOG_BYMINUTE
where 1=1
	and COLLECT_DATE >= 20190915
group by
	DEVICE_ID
	, COLLECT_DATE


select 	
	device_id
	, count(DISTINCT APPLIANCE_NO)
	, count(DISTINCT APPLIANCE_TYPE)
from(	SELECT
		    t2.DEVICE_ID
		    , t1.APPLIANCE_NO
		    , t1.APPLIANCE_TYPE
		FROM AH_APPLIANCE t1
		INNER JOIN 
			(select *
			from AH_APPLIANCE_CONNECT
			where 1=1
			and FLAG_DELETE = 'N'
			) t2
		ON t1.APPLIANCE_NO = t2.APPLIANCE_NO) tt
group by DEVICE_ID


SELECT
	APPLIANCE_TYPE
	, count(*)
from (	SELECT
		    t2.DEVICE_ID
		    , t1.APPLIANCE_NO
		    , t1.APPLIANCE_TYPE
		FROM AH_APPLIANCE t1
		INNER JOIN 
			(select *
			from AH_APPLIANCE_CONNECT
			where 1=1
			and FLAG_DELETE = 'N'
			) t2
		ON t1.APPLIANCE_NO = t2.APPLIANCE_NO) s
group by APPLIANCE_TYPE
		