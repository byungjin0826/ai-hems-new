import utils
import pandas as pd
import pymysql

db = 'silver_service_db'

# db connection 정보.
conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                            port=3306, user='aihems', passwd='#cslee1234', db=db,
                                            charset='utf8')
# sql문
sql = f"""
SELECT *
FROM SC_SAMPLE
"""

df = pd.read_sql(sql, conn)
df = df.rename(str.lower, axis='columns')

# label_status:3
# avg_status:4
# danger:5
# special_moment:6


avgWhating = int(0)
status = int(0)
danger = int(0)
special_moment = int(0)

for i in range(len(df)):
# for i in range(2000):
    if df.iloc[i, 3] != df.iloc[i, 4]:
        if df.iloc[i, 4]  == 1:
            danger += 1
            df.iloc[i, 5] = danger
            df.iloc[i, 6] = special_moment
            # df.iloc[i, 6] = 0
        else:
            # df.iloc[i, 5] = 0
            special_moment += 1
            df.iloc[i, 5] = danger
            df.iloc[i, 6] = special_moment

    if df.iloc[i, 3] != df.iloc[i - 1, 3]:
            danger = 0
            special_moment = 0
            status = 0
            df.iloc[i, 5] = danger
            df.iloc[i, 6] = special_moment

    else:
        if df.iloc[i, 3] == df.iloc[i-1, 3]:
            status == status
            df.iloc[i, 5] = danger
            df.iloc[i, 6] = special_moment

        elif df.iloc[i, 3] == df.iloc[i, 4]:
            danger = 0
            special_moment = 0
            status = 0
print(1)

cursor = conn.cursor()

i = 0
for row in df.iterrows():
    house_no = row[1]['house_no']
    collect_time = row[1]['collect_time']
    danger = row[1]['danger']
    special_moment = row[1]['special_moment']

    sql = f"""
update SC_SAMPLE
set danger = {danger}, special_moment = {special_moment}
where 1=1
and house_no = '{house_no}'
and collect_time = '{collect_time}'
"""

    cursor.execute(sql)
    conn.commit()
    i += 1
    print(i)

