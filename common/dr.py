import pandas as pd
import settings


def cbl_info(house_no, request_dr_no):
    sql = f"""
SELECT sum(s)/4
from (	SELECT *
        FROM (	SELECT 
                    sum(ENERGY_DIFF) s
                FROM AH_USE_LOG_BYMINUTE
                WHERE 1=1
                AND GATEWAY_ID = (
                    SELECT GATEWAY_ID
                    FROM AH_GATEWAY_INSTALL
                    WHERE 1=1
                    AND HOUSE_NO = '{house_no}'
                )
                AND COLLECT_DATE >= DATE_FORMAT(DATE_ADD((SELECT START_DATE FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}'), INTERVAL -5 DAY), '%Y%m%d')
                AND COLLECT_DATE < (SELECT DATE_FORMAT(START_DATE, '%Y%m%d') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
                AND COLLECT_TIME >= (SELECT DATE_FORMAT(START_DATE, '%H%i') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
                AND COLLECT_TIME <= (SELECT DATE_FORMAT(END_DATE, '%H%i') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
                GROUP BY
                    COLLECT_DATE) t1
        WHERE 1=1
        ORDER BY s desc
        limit 4) t2"""

    cbl = pd.read_sql(sql, con=settings.conn).iloc[0, 0]

    if cbl <= 500:
        reduction_energy = cbl * 0.3
    elif cbl <= 1500:
        reduction_energy = cbl * 0.15 + 75
    else:
        reduction_energy = 300

    return (cbl, reduction_energy)


if __name__ == '__main__':
    print('hello')
