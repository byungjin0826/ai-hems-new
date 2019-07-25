import utils


gateway_id = 'ep18270363'
device_id = '000D6F0012577C7C1'

sql = f"""
SELECT *
FROM AH_USE_LOG_BYMINUTE_201904
WHERE 1=1
AND GATEWAY_ID = '{gateway_id}'
AND DEVICE_ID = '{device_id}'
"""

df = utils.get_table_from_db(sql)
df = utils.binding_time(df)

schedule = df.pivot_table(values='appliance_status', index=df.index.time, columns=df.index.dayofweek,
                          aggfunc='max')

schedule = schedule.reset_index()

schedule_unpivoted = schedule.melt(id_vars=['index'], var_name='date', value_name='appliance_status')

schedule_unpivoted.loc[:, 'status_change'] = schedule_unpivoted.appliance_status == schedule_unpivoted.appliance_status.shift(1)

subset = schedule_unpivoted.loc[
    (schedule_unpivoted.status_change == False), ['date', 'index', 'appliance_status']]

subset.columns = ['dayofweek', 'time', 'appliance_status']

subset.loc[:, 'minutes'] = [x.hour * 60 + x.minute for x in subset.time]

subset.loc[:, 'minutes'] = subset.dayofweek * 1440 + subset.minutes

subset.loc[:, 'duration'] = subset.minutes - subset.minutes.shift(1)
subset.loc[:, 'duration'] = subset.minutes.shift(-1) - subset.minutes

subset = subset.loc[((subset.appliance_status == 0) & (subset.duration < 120)) == False, :]
# subset = subset.loc[subset.duration > 120, :]

subset.loc[:, 'status_change'] = subset.appliance_status == subset.appliance_status.shift(1)

subset = subset.loc[(subset.status_change == False), :]

subset.loc[:, 'dayofweek'] = [str(x) for x in subset.loc[:, 'dayofweek']]

subset.loc[:, 'time'] = [str(x) for x in subset.loc[:, 'time']]

subset = subset.reset_index(drop=True)

result = subset.to_dict('index')