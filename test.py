import utils


gateway_id = 'ep18270363'
device_id = '000D6F001257E60C1'
# dayofweek = 0

sql = f"""
SELECT *
FROM AH_USE_LOG_BYMINUTE_201904
WHERE 1=1
AND GATEWAY_ID = '{gateway_id}'
AND DEVICE_ID = '{device_id}'            
"""

df = utils.get_table_from_db(sql)
df = utils.binding_time(df)

schedule = df.pivot_table(values='appliance_status', index=df.index.time, columns=df.index.dayofweek, aggfunc='max')

schedule = schedule.reset_index()

schedule_unpivoted = schedule.melt(id_vars=['index'], var_name='date', value_name='appliance_status')

schedule_unpivoted.loc[:, 'status_change'] = schedule_unpivoted.appliance_status == schedule_unpivoted.appliance_status.shift(1)

subset = schedule_unpivoted.loc[(schedule_unpivoted.status_change == False), ['date', 'index', 'appliance_status']]

subset.columns = ['dayofweek', 'time','appliance_status']

subset.loc[:, 'date'] = [str(x) for x in subset.loc[:, 'date']]

subset.loc[:, 'time'] = [str(x) for x in subset.loc[:, 'time']]

subset = subset.reset_index(drop=True)

# df_unpivoted = df.melt(id_vars=['car_model'], var_name='date', value_name='0-60mph_in_seconds')
# df_unpivoted

result = subset.to_dict('index')