import pymysql
from sqlalchemy import create_engine
import pandas as pd

def data_load(member_name, appliance_name, months = None):
    months = months or None
    encoding = 'euc-kr'
    # df1 = pd.read_csv('./sample_data/csv/aihems/' + appliance_name + '(' + member_name + ')_01.csv',
    #                   encoding=encoding)  # 24일 이전 데이터 x
    # if months != 1:
    #     df2 = pd.read_csv('./sample_data/csv/aihems/' + appliance_name + '(' + member_name + ')_02.csv',
    #                       encoding=encoding)  # 2일부터...
    #     df1 = pd.concat([df1, df2], ignore_index=True)
    df1 = pd.read_csv('./sample_data/csv/aihems/' + appliance_name + '(' + member_name + ').csv',
                      encoding=encoding) #실증세대
    df1 = df1.loc[df1.energy != '\\N', :].copy()
    df1.columns.values[-1] = 'appliance_status'  # excel에 컬럼값 입력 안됨
    return(df1)

def set_data(df, source = None):
    source = source or None

    if source == 'excel':
        df1 = df.loc[df.energy != '\\N', :].copy()  # db에서 load 할 때는 na로 들어옴.

    else:
        df1 = df.dropna()
        df1.loc[:, 'appliance_status'] = 0

    df1.loc[:, 'collected_time'] = [str(x).rjust(4, '0') for x in df1.collected_time]
    df1.loc[:, 'collected_time'] = [x[:2] + ':' + x[2:] for x in df1.collected_time]
    df1.loc[:, 'collected_date'] = [str(x) for x in df1.collected_date]
    df1.loc[:, 'date_time'] = df1.loc[:, 'collected_date'] + ' ' + df1.loc[:, 'collected_time']
    df1.loc[:, 'date_time'] = pd.to_datetime(df1.loc[:, 'date_time'])

    df1.loc[:, 'smart_plug_onoff'] = df1.loc[:, 'onoff']

    df1.loc[:, 'smart_plug_onoff'] = [int(x) for x in df1.smart_plug_onoff]
    df1.loc[df1.appliance_status.isna(), 'appliance_status'] = 0
    df1.loc[:, 'appliance_status'] = [int(x) for x in df1.appliance_status]

    df1.loc[:, 'date'] = [str(x.date()) for x in df1.date_time]
    df1.loc[:, 'dayofweek'] = [str(x.dayofweek) for x in df1.date_time]  # 0이 월요일

    df1.loc[:, 'time'] = [str(x.time()) for x in df1.date_time]
    df1.loc[:, 'month'] = [str(x.month) for x in df1.date_time]
    df1.loc[:, 'day'] = [str(x.day) for x in df1.date_time]
    df1.loc[:, 'hour'] = [int(x.hour) for x in df1.date_time]
    df1.hour = [[x] for x in df1.hour]
    df1.loc[:, 'minute'] = [[int(x.minute)] for x in df1.date_time]
    df1.minute= [[x] for x in df1.minute]

    df1.loc[:, 'dayofyear'] = [int(x.dayofyear) for x in df1.date_time]

    df1.loc[:, 'energy'] = [float(x) for x in df1.energy]
    df1.loc[:, 'energy_lagged'] = df1.energy.shift(+1)
    df1.loc[:, 'energy_diff'] = df1.energy - df1.energy_lagged
    df1.iloc[-1, 8] = 0
    df1.energy_diff[0] = 0

    # df1.loc[:, 'gateway_id'] = 'ep17470141'
    df1.loc[:, 'end_point'] = 1
    df1.loc[:, 'quality'] = 100
    df1.loc[:, 'create_date'] = pd.datetime.today()
    gateway_id = df1.gateway_id[0]
    df1.gateway_id = gateway_id[:6] + gateway_id[-4:]

    return (df1)

def transform_data(df):
    df = df.loc[:, [
          'gateway_id' #
        , 'device_address' #
        , 'end_point'
        , 'collected_date' #
        , 'collected_time' #
        , 'quality'
        , 'onoff' #
        , 'energy' #
        , 'energy_diff' #
        , 'appliance_status' #
        , 'create_date'
                  ]]
    return(df)
def write_db(df):
    user = 'aihems'
    passwd = '#cslee1234'
    addr = 'aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com'
    port = "3306"
    db_name = 'aihems_api_db'
    engine = create_engine("mysql+mysqldb://"+user+":" + passwd +"@"+addr+":"+port+"/"+db_name,
                           encoding='utf-8')
    # conn = engine.connect()
    df.to_sql('AH_USE_LOG_BYMINUTE_LABLED', con=engine, if_exists='append', index=False)
    return(0)
member_list = [
                # ['안준필','dyson청소기']
                # , ['이영복','PC']
                # , ['이철희','PC-HOME']
                # , ['이철희','PC-건너방']
                # , ['안준필','Samsung TV']
                # , ['심준','TV(LG_OLED_201511']Q
                # , ['박선주','TV(SKYMEDIA 43)']
                # , ['김성수','TV']
                # , ['김이래','TV']
                # , ['신병진','TV']
                # , ['신용건','TV']
                # , ['이길무','TV']
                # , ['이병노','TV']
                # , ['신병진','가스레인지 후드']
                # , ['이철희','거실TV']
                # , ['신병진','노트북 충전기']
                # , ['김성수','노트북']
                # , ['신병진','드라이기']
                # , ['김지명','로봇청소기']
                # , ['박재훈','로봇청소기']
                # , ['박선주','맥북에어(13)']
                # , ['김성수','모니터']
                # , ['김성수','무선청소기']
                # , ['신병진','믹서기']
                # , ['석효천','밥솥']
                # , ['박재훈','비데']
                # , ['김이래','선풍기']
                # , ['박재훈','세탁기']
                # , ['신용건','세탁기']
                # , ['김성수','오디오']
                # , ['신용건','오디오']
                # , ['이철희','장기장판2(작은방)']
                # # , ['박재훈','전기밥솥'] E
                # # , ['신병진','전기밥솥'] E
                # , ['신용건','전기밥솥']
                # , ['이영복','전기밥솥']
                # , ['박선주','전기밥솥(쿠쿠 CRP-HMXT1070SB)']
                # , ['박재훈','전자레인지']
                # # , ['신병진','전자레인지'] E
                # , ['신용건','전자레인지']
                # , ['김이래','제습기']
                # , ['이영복','충전기']
                # , ['윤희우','컴퓨터']
                # , ['이철희','프린터']
                # , ['신병진','핸드폰 충전기']
                # , ['김이래','헤어드라이기']
]

member_name = input('사용자 이름: ')
appliance_name = input('가전기기 이름: ')
#
# for i in member_list:
#     member_name = i[0]
#     appliance_name = i[1]
#     df = data_load(member_name = member_name, appliance_name = appliance_name)
#     df = set_data(df, source = 'excel')
#     df1 = transform_data(df)
#
#     write_db(df1)

#member_name = i[0]
#appliance_name = i[1]
df = data_load(member_name = member_name, appliance_name = appliance_name)
df = set_data(df, source = 'excel')
df1 = transform_data(df)

write_db(df1)
