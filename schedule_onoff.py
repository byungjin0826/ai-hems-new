import pandas as pd
import numpy as np
import time

# todo: 시계열 데이터 빈 값 처리 방법
# https://www.kaggle.com/juejuewang/handle-missing-values-in-time-series-for-beginners
# todo: index를 MultiIndex로 구성(date_time, appliance)
# https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
# todo: appliance와 plug를 분리

# 데이터 로드
# 없는 데이터 삭제....-> 비어있는 데이터를 어떻게 처리할 것인지...
# collected_date / collected_time: 두 개의 컬럼을 datetime 형태로 변환
# 휴일인 경우는 제외
# pivot 해서 나온 결과와 appliance_off랑 비교

start = time.time()  # 시간 측정 시작
def set_data(df):
    df = df.loc[df.energy != '\\N', :]

    df.loc[:, 'collected_time'] = [str(x).rjust(4, '0') for x in df.collected_time]
    df.loc[:, 'collected_time'] = [x[:2]+':'+x[2:] for x in df.collected_time]
    df.loc[:, 'collected_date'] = [str(x) for x in df.collected_date]
    df.loc[:, 'date_time'] = df.loc[:, 'collected_date'] + ' ' + df.loc[:, 'collected_time']
    df.loc[:, 'date_time'] = pd.to_datetime(df.loc[:, 'date_time'])
    df.loc[:, 'smart_plug_onoff'] = df.loc[:, 'onoff']

    df.loc[:, 'smart_plug_onoff'] = [int(x) for x in df.smart_plug_onoff]
    df.loc[:, 'appliance_onoff'] = [int(x) for x in df.appliance_onoff]

    df.loc[:, 'energy'] = [float(x) for x in df.energy]
    df.loc[:, 'energy_lagged'] = df.energy.shift(-1)
    df.loc[:, 'energy_diff'] = df.energy_lagged - df.energy
    df.iloc[-1, 8] = 0

    df = df.loc[:, ['date_time', 'energy', 'energy_diff', 'smart_plug_onoff', 'appliance_onoff']]
    return(df)

def make_schedule(df):
    df.loc[:, 'time'] = [str(x.time()) for x in  df.date_time]
    df.loc[:, 'day_of_week'] = [str(x.dayofweek) for x in df.date_time]
    pivotted_df = pd.pivot_table(df, index = 'time', columns = 'day_of_week',
                                 values = 'appliance_onoff', aggfunc= max)
    return(pivotted_df)

def reverse_pivot_table(pivotted_df):
    pivotted_df = pivotted_df.reset_index()
    reversed_pivot_table = pd.melt(pivotted_df, id_vars = 'time', value_vars = list(pivotted_df.columns[1:]),
                                   var_name = 'day_of_week', value_name = 'appliance_onoff_predicted')
    return(reversed_pivot_table)

def merge_table(df, reversed_pivot_table):
    df = pd.merge(df, reversed_pivot_table, how = 'left', on = ['time', 'day_of_week'])
    return(df)

# 1월, 2월 데이터 로드
encoding = 'euc-kr'
df1 = pd.read_csv('./sample_data/csv/세탁기(박재훈)_01.csv', encoding = encoding) # 24일 이전 데이터 x
df2 = pd.read_csv('./sample_data/csv/세탁기(박재훈)_02.csv', encoding = encoding) # 2일부터...
df = pd.concat([df1, df2], ignore_index = True)
df.columns.values[-1] = 'appliance_onoff' # excel에 컬럼값 입력 안됨

set_start = time.time() # 시간 측정 시작
df = set_data(df)
set_end = time.time() # 시간 측정 종료
print('데이터 전처리 시간: ', round(set_end - set_start, 3),'s', sep = "")

# todo: 3주씩 단위로 바꿔서 해보기
pivotted_df = make_schedule(df)
reversed_pivot_table = reverse_pivot_table(pivotted_df)
merged_df = merge_table(df, reversed_pivot_table)

end = time.time() # 시간 측정 종료
print('총 수행시간: ', round(float(end-start), 4),'s',sep = "")

# 대기전력 산출
# type 1 error: appliance_onoff = 0 & smart_plug_onoff = 1
# type 2 error: appliance_onoff = 1 & smart_plug_onoff = 0
print('=======================================')
print('최대 대기전력 절감량: ',
      round(sum(merged_df.loc[(merged_df.smart_plug_onoff == 1) & (merged_df.appliance_onoff == 0), 'energy_diff']),3),
      'kWh', sep = "")

# print('type 1 error: ',
#       round(sum(merged_df.loc[(merged_df.smart_plug_onoff == 1) & (merged_df.appliance_onoff_predicted == 0), 'energy_diff']), 3),
#       'kWh', sep="")
# print('type 2 error: ',
#       round(sum(merged_df.loc[(merged_df.smart_plug_onoff == 0) & (merged_df.appliance_onoff_predicted == 1), 'energy_diff']), 3),
#       'kWh', sep="")

df_type_1_error = merged_df.loc[(merged_df.smart_plug_onoff == 1) & (merged_df.appliance_onoff_predicted == 0), :]
print('절감 실패: ',
      len(merged_df.loc[(merged_df.appliance_onoff == 0) & (merged_df.appliance_onoff_predicted == 1), 'energy_diff']),
      'min, ',
      round(sum(merged_df.loc[(merged_df.appliance_onoff == 0) & (merged_df.appliance_onoff_predicted == 1), 'energy_diff']),3),
      'kWh',
      sep = "")
print('사용자 불편: ',
      len(merged_df.loc[(merged_df.appliance_onoff == 1) & (merged_df.appliance_onoff_predicted == 0), 'energy_diff']),
      'min',
      sep="")

