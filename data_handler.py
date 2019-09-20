##using data : COLLECT_TIME, COLLECT_DATE, ENERGY_DIFF, ONOFF_TV

# import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pymysql
import math
from scipy import ravel

class TodayDate:
    def __init__(self):
        super().__init__()

    def today(self):
        return int(datetime.datetime.now().strftime('%Y%m%d'))


class GetData:
    data = pd.DataFrame
    nec_data = pd.DataFrame
    def __init__(self, gateway, input_date):
        self._gateway = gateway
        self._input_date = input_date
        GetData.data = self.data_road()
        GetData.nec_data = self.necessary_data()
        super().__init__()

    def data_road(self):
        gateway = self._gateway
        engine = create_engine(
            "mysql+pymysql://aihems:" + "#cslee1234" +
            "@aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com:3306/aihems_service_db?charset=utf8",
            encoding='utf-8')
        conn = engine.connect()
        data = pd.read_sql('SELECT * ' +
                           'FROM AH_LOG_SOCKET '+
                           'WHERE GATEWAY_ID=' + "'" + gateway + "'", con=conn)

        conn.close()
        return data

    def necessary_data(self):
        input_date = self._input_date
        data = GetData.data
        # data road
#         data = self.data_road()
        data['ENERGY_DIFF'] = data['ENERGY'] - data['ENERGY'].shift(periods=3, fill_value=0)
        data['ENERGY_DIFF_'] = data['ENERGY'] - data['ENERGY'].shift(periods=-3, fill_value=0)
        data.iloc[-3:, 12] = data.iloc[-4, 12]
        
        data = data.loc[
            data['COLLECT_DATE'].astype(int) <= int(input_date), ['COLLECT_DATE', 'COLLECT_TIME', 'ENERGY_DIFF', 'ENERGY_DIFF_']]

        nec_data = data.iloc[-20160:]  # 7일치 데이터
        nec_data.loc[:,'ONOFF_TV'] = 1
        nec_data.loc[nec_data['ENERGY_DIFF'] == 0, 'ONOFF_TV'] = 0
        nec_data.loc[nec_data['ENERGY_DIFF_'] == 0, 'ONOFF_TV'] = 0

        nec_data = nec_data.reset_index(drop=True)

        nec_data['TIME_BAND'] = nec_data['COLLECT_TIME'].astype(int) + (
                30 - nec_data['COLLECT_TIME'].astype(int) % 100 % 30).astype(int)  # 30분단위 시간
        nec_data.loc[nec_data['TIME_BAND'] % 100 >= 60, 'TIME_BAND'] += 40

#         for data in nec_data['TIME_BAND']:x
#             if data % 100 >= 60:
#                 nec_data.loc[nec_data['TIME_BAND'] == data, 'TIME_BAND'] += 40

        nec_data['WEIGHT'] = (nec_data.index / 1440).astype(int) + 20  # 가중치
        nec_data['GIVE_WEIGHT_ONOFF'] = nec_data['WEIGHT'] * nec_data['ONOFF_TV']
          # 가중치 * TV사용여부

        return nec_data

    def use_by_time(self): # 하루마다 갱신
        input_date = self._input_date
        nec_data = GetData.nec_data
        # using_by_time = nec_data[['TIME_BAND', 'ONOFF_TV']].groupby(['TIME_BAND']).mean()
        using_by_time = nec_data.loc[nec_data['COLLECT_DATE'].astype(int) < int(input_date)]
        using_by_time = using_by_time[['TIME_BAND', 'GIVE_WEIGHT_ONOFF']].groupby(['TIME_BAND']).mean()  # 어제까지의 평균
        # print(using_by_time)
        using_by_time['GIVE_WEIGHT_ONOFF'] = using_by_time['GIVE_WEIGHT_ONOFF'] / 34.5
        return using_by_time

    def use_today(self): # 30분마다 갱신
        input_date = self._input_date
        nec_data = GetData.nec_data
        using_today = nec_data.loc[nec_data['COLLECT_DATE'].astype(int) == int(input_date)]
        using_today = using_today[['TIME_BAND', 'ONOFF_TV']].groupby(['TIME_BAND']).mean()  # 오늘 사용시간
        using_today.loc[using_today['ONOFF_TV'] <= 0.1, 'ONOFF_TV'] = 0
        using_today.loc[using_today['ONOFF_TV'] >= 0.9, 'ONOFF_TV'] = 1
        return using_today

    def use_24hours(self): # 30분마다 갱신
        nec_data = GetData.nec_data
        # 최근24시간 사용량 (이탈률 계산을 위해)
        using_24hours = nec_data.iloc[-1440:]
        using_24hours = using_24hours[['TIME_BAND', 'ONOFF_TV']].groupby(['TIME_BAND']).mean()
        using_24hours.loc[using_24hours['ONOFF_TV'] <= 0.1, 'ONOFF_TV'] = 0
        using_24hours.loc[using_24hours['ONOFF_TV'] >= 0.9, 'ONOFF_TV'] = 1
        return using_24hours

    def bounce_rate(self):
        using_by_time = self.use_by_time()
        using_24hours = self.use_24hours()
        bounce_rate = pd.DataFrame(abs(using_by_time['GIVE_WEIGHT_ONOFF'] - using_24hours['ONOFF_TV']),
                                   columns=["bounce"])  # 이탈률

        bounce_rate['bounce_sum'] = 0.0
        for data in range(bounce_rate['bounce'].size):
            bounce_rate.iat[data, 1] = bounce_rate.iloc[0:data + 1, 0].sum()  # 누적이탈률
        return bounce_rate.iat[-1, 1]
    
    def bounce_rate_store_all(self, date): # 이탈률 어제까지 전체 저장
#         input_date = self._input_date
        gateway = self._gateway
        
        conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com', port=3306, user='aihems', password='#cslee1234', database='silver_service_db')  
        cursor = conn.cursor()
        dt_index = pd.date_range(start=date, end=datetime.datetime.now().strftime('%Y%m%d'), closed='left')
        dt_list = dt_index.strftime("%Y%m%d").tolist()
        for i in dt_list: #고쳐야함##############3
            i = int(i)
            sql ='DELETE FROM BOUNCE_RATE WHERE GATEWAY_ID=' + "'" + gateway + "'" + ' AND COLLECT_DATE=' + "'" + str(i) + "'"
            cursor.execute(sql)
            sql ='INSERT INTO BOUNCE_RATE VALUES (' + "'" + gateway + "'" + ', ' + str(i) + ', ' + str(GetData(gateway, i).bounce_rate()) + ')'
            cursor.execute(sql)

        conn.commit()
        conn.close()
        
    def bounce_rate_store(self): # 이탈률 저장, 하루마다 갱신
        gateway = self._gateway
        day = int((datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')) #고쳐야함
        conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com', port=3306, user='aihems', password='#cslee1234', database='silver_service_db')  
        cursor = conn.cursor()
        sql = 'DELETE FROM BOUNCE_RATE WHERE GATEWAY_ID=' + "'" + gateway + "'" + ' AND COLLECT_DATE=' + "'" + str(day) + "'"
        cursor.execute(sql)
        sql = 'INSERT INTO BOUNCE_RATE VALUES (' + "'" + gateway + "'" + ', ' + str(day) + ', ' + str(GetData(gateway, day).bounce_rate()) + ')'
        cursor.execute(sql)

        conn.commit()
        conn.close()
    
    def change_show_data(self):
        using_by_time = self.use_by_time()
        using_today = self.use_today()
        show_data = pd.DataFrame(columns=['AVG', 'TODAY'])
        show_data['AVG'] = using_by_time['GIVE_WEIGHT_ONOFF']
        show_data['TODAY'] = using_today['ONOFF_TV']
        show_data.loc[show_data['AVG'] < 0.5, 'AVG'] = 0
        show_data.loc[show_data['AVG'] >= 0.5, 'AVG'] = 1
        show_data.loc[show_data['TODAY'] < 0.5, 'TODAY'] = 0
        show_data.loc[show_data['TODAY'] >= 0.5, 'TODAY'] = 1
        return show_data

    def show(self): # 오늘과 평균의 그래프 비교
        using_by_time = self.use_by_time()
        using_today = self.use_today()
        predict = self.predict_by_ml()
        using_by_time.loc[using_by_time['GIVE_WEIGHT_ONOFF'] < 0.5, 'GIVE_WEIGHT_ONOFF'] = 0
        using_by_time.loc[using_by_time['GIVE_WEIGHT_ONOFF'] >= 0.5, 'GIVE_WEIGHT_ONOFF'] = 1
        using_today.loc[using_today['ONOFF_TV'] < 0.5, 'ONOFF_TV'] = 0
        using_today.loc[using_today['ONOFF_TV'] >= 0.5, 'ONOFF_TV'] = 1
        plt.figure()
#         a = pd.concat([using_today, predict], axis=1)
#         a.plot.bar(rot=0)
        plt.subplot(3, 1, 1)
        plt.plot(using_by_time, label='AVG', color='red')
        plt.title('AVG')
        plt.xlabel('time')
        plt.ylabel('usingTV')
        
        plt.subplot(3, 1, 2)
        plt.plot(using_today, label='today', color='blue')
        plt.title('today')
        plt.xlabel('time')
        plt.ylabel('usingTV')
        
        plt.subplot(3, 1, 3)
        plt.plot(predict, label='predict', color='green')
        plt.title('predict')
        plt.xlabel('time')
        plt.ylabel('usingTV')
        
        plt.tight_layout()
        plt.legend()
        plt.show()
        
            
    def predict_by_ml(self): #머신러닝 예측
        input_date = self._input_date
        nec_data = GetData.nec_data
            
        train_data = nec_data[['COLLECT_DATE', 'COLLECT_TIME', 'ONOFF_TV', 'TIME_BAND']]
        
        train_data = train_data.loc[train_data['COLLECT_DATE'].astype(int) < int(input_date), ['COLLECT_DATE', 'COLLECT_TIME', 'ONOFF_TV', 'TIME_BAND']]
        
        train_data['DAY_OF_WEEK'] = pd.to_datetime(train_data['COLLECT_DATE']).dt.dayofweek
        
        train_data.loc[:, ['COLLECT_DATE', 'COLLECT_TIME', 'TIME_BAND', 'DAY_OF_WEEK', 'ONOFF_TV']]
#         train_data = train_data.groupby([train_data['COLLECT_DATE'], train_data['TIME_BAND']], as_index=False).mean()
        
        x_train = train_data.loc[:, ['COLLECT_DATE', 'COLLECT_TIME', 'TIME_BAND', 'DAY_OF_WEEK']]
#         x_train[['COLLECT_DATE', 'DAY_OF_WEEK']] = x_train[['COLLECT_DATE', 'DAY_OF_WEEK']].astype(int)
        y_train = train_data.loc[:, ['ONOFF_TV']]
#         y_train.loc[y_train['ONOFF_TV'] <= 0.1, 'ONOFF_TV'] = 0
#         y_train.loc[y_train['ONOFF_TV'] >= 0.9, 'ONOFF_TV'] = 1
#         print(x_train)
#         print(y_train)
#         print(x_train)
#         print(y_train)
        x_test = pd.DataFrame(columns=['COLLECT_DATE', 'COLLECT_TIME', 'TIME_BAND', 'DAY_OF_WEEK'], index=range(0, 1440))
        x_test.loc[:, 'COLLECT_DATE'] = input_date
        x_test.loc[:, 'COLLECT_TIME'] = x_test.index % 60 + (x_test.index / 60).astype(int) * 100
        x_test.loc[:, 'TIME_BAND'] = (((x_test.index / 30).astype(int) * 30 + 30) / 60).astype(int) * 100 + (((x_test.index / 30).astype(int) * 30 + 30) % 60)
        x_test.loc[:, 'DAY_OF_WEEK'] = datetime.datetime.strptime(str(input_date), "%Y%m%d").date().weekday()
#         print(x_test)
#         print(type(input_date))
        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train, y_train.values.ravel())
        prediction = model.predict(x_test)
        prediction = pd.DataFrame(prediction, columns=['ONOFF_TV'])
#         print(prediction)
        
        result = pd.concat([x_test, prediction], axis=1)
        result = result[['TIME_BAND', 'ONOFF_TV']].groupby(['TIME_BAND']).mean()
        result.loc[result['ONOFF_TV'] < 0.5, 'ONOFF_TV'] = 0
        result.loc[result['ONOFF_TV'] >= 0.5, 'ONOFF_TV'] = 1
        return result
     
#         x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=5)
#         model.fit(x_train, y_train.astype(int))
#         prediction = model.predict(x_test)
#         accuracy = round(accuracy_score(y_test.astype(int), prediction) * 100, 2)
#         print("Accuracy : ", accuracy, "%")
#         print(prediction)
    
    
    
# f = GetData('ep18270153', 20190823)
# print(f.predict_by_ml())
# print(str(f.predict_by_ml().loc[int('0300'), 'ONOFF_TV']))
# print(f.use_today())
# print(f.use_today().loc[int('0100'), 'ONOFF_TV'])
# f.show()

class CompareBounce:
    bounce_lists = pd.DataFrame
    def __init__(self, gateway, input_date):
        self._gateway = gateway
        self._input_date = input_date
        CompareBounce.bounce_lists = self.bounce_list()
        super().__init__()

    def bounce_list(self): # 지난 6일간의 이탈률 list -> DB에서 꺼내오는거로 변환
#         bounce_list = []
#         gateway = self._gateway
#         input_date = self._input_date
        gateway = self._gateway
        engine = create_engine(
            "mysql+pymysql://aihems:" + "#cslee1234" +
            "@aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com:3306/silver_service_db?charset=utf8",
            encoding='utf-8')
        conn = engine.connect()
  
        bounce_lists = pd.read_sql('select BOUNCE_RATE from BOUNCE_RATE ' + 'WHERE COLLECT_DATE >= ' +
                                   (datetime.datetime.now() - relativedelta(months=1)).strftime('%Y%m%d') + ' AND GATEWAY_ID = ' + "'" + gateway + "'", con=conn)
        conn.close()
        
#         for i in range(input_date - 6, input_date):
#             bounce_list.append(GetData(gateway, i).bounce_rate())
        return bounce_lists
        
    def mean_bounce(self):
        bounce_lists = CompareBounce.bounce_lists
        return bounce_lists.mean(axis=0, skipna = True).values[0]

    def iqr_bounce(self):
        bounce_lists = CompareBounce.bounce_lists
        q1 = bounce_lists.quantile(0.25)
        q3 = bounce_lists.quantile(0.75)
        iqr = q3 - q1
        return iqr.values[0]

    def alarm(self): # 30분마다 갱신
        gateway = self._gateway
        input_date = self._input_date
        bounce_lists = CompareBounce.bounce_lists
#         median_bounce = self.median_bounce()
#         iqr_bounce = self.iqr_bounce()
        
        q1 = bounce_lists.quantile(0.25).values[0]
        q3 = bounce_lists.quantile(0.75).values[0]
        iqr = q3 - q1
        
#         print('iqr계산', q1, q3, iqr)
        
        f = GetData(gateway, input_date).bounce_rate()
        if f > q3 + iqr * 3 or f < q1 -iqr * 3:
            return 3
        elif f > q3 + iqr * 1.5 or f < q1 -iqr * 1.5:
            return 2
        else:
            return 1

# f = GetData('ep18270153', '20190813')        
# f2 = CompareBounce('ep18270153', '20190813')
# print(f2.bounce_list)
# print(type(f2.bounce_list))
# print(f.nec_data)
# print(f.nec_data['COLLECT_DATE'])
# t = TodayDate().today()
# f = GetData('ep18270185', 20190817)
# f.show()
# f = GetData('ep18270161', 20190819)
# print(f.necessary_data())
# f2 = GetData('ep18270185', 20190818)
# f3 = GetData('ep18270185', 20190817)
# f4 = GetData('ep18270185', 20190816)
# f5 = GetData('ep18270185', 20190815)
# f6 = GetData('ep18270185', 20190814)
# f6 = GetData('ep18270185', 20190813)
#
# print(f.bounce_rate())

# print(GetData('ep18270185', 20190819).bounce_rate())
# print(CompareBounce('ep18270185', 20190819).bounce_list())
# g = CompareBounce('ep18270185', 20190819)
# g.alarm()
# print(g.mean_bounce())
# print(g.std_bounce())

if __name__ == '__main__':
