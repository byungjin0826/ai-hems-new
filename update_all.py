'''
Created on 2019. 8. 20.

@author: user
'''

############## 초기 데이터 설정 ##################
import data_handler as dh
import pandas as pd
import datetime
from sqlalchemy import create_engine
import pymysql

def gateway_list():
    engine = create_engine(
        "mysql+pymysql://aihems:" + "#cslee1234" +
        "@aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com:3306/silver_service_db?charset=utf8",
        encoding='utf-8')
    conn = engine.connect()
    gateway_list = pd.read_sql('SELECT gateway_id FROM HOUSE_DATA', con=conn)
    conn.close()
    return gateway_list

if __name__ == '__main__':
    gateway_list = gateway_list()
    time_band = ['0030', '0100', '0130', '0200', '0230', '0300', '0330', '0400', '0430', '0500', '0530', '0600', '0630', '0700', '0730', '0800', '0830', 
                 '0900', '0930', '1000', '1030', '1100', '1130', '1200', '1230', '1300', '1330', '1400', '1430', '1500', '1530', '1600', 
                 '1630', '1700', '1730', '1800', '1830', '1900', '1930', '2000', '2030', '2100', '2130', '2200', '2230', '2300', '2330', '2400']
    today = dh.TodayDate().today()
    
    conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com', port=3306, user='aihems', password='#cslee1234', database='silver_service_db')  
    cursor = conn.cursor()
    
    dt_index = pd.date_range(start='2019-08-23', end=datetime.datetime.now())
    dt_list = dt_index.strftime("%Y%m%d").tolist()
    
    cur = int(datetime.datetime.now().strftime('%H%M'))  - (int(datetime.datetime.now().strftime('%H%M')) % 100 % 30)
                
    if cur%100 >= 60:
        cur += 40
    else:
        pass
                
    if cur < 100:
        cur = '00'+ str(cur)
    elif cur < 1000:
        cur = '0' + str(cur)
    else:
        cur = str(cur)
#     f = dh.GetData('ep18270154', 20190820)
#     print(f.use_by_time().loc[100, 'GIVE_WEIGHT_ONOFF'])
#     print(f.use_by_time())
#     print(f.use_today().loc[0, 'ONOFF_TV'])
#     print(f.use_today().loc[100, 'ONOFF_TV'])
    

    for i in range(0, 30):
        gateway = gateway_list.values[i][0]
        if gateway == 'ep18270158' or gateway == 'ep18270186' or gateway == 'ep18270394' or gateway == 'ep18270397' or gateway == 'ep18270408':
            pass
        else:
            a = dh.GetData(gateway, today) #(게이트웨이 번호,날짜)
            a.bounce_rate_store_all('20190823')  #어제까지의 총 이탈률 DB에 한꺼번에 저장 # 한번만 사용
            for j in dt_list: #고쳐야함##############3
                j = int(j)
                f = dh.GetData(gateway, j)
                f2 = dh.CompareBounce(gateway, j)
                if (f.nec_data['COLLECT_DATE']==str(j)).any():
                    use_by_time_ = f.use_by_time()
                    use_today_ = f.use_today()
                    bounce_rate_ = f.bounce_rate()
                    alarm_ = f2.alarm()
                    mean_bounce_ = f2.mean_bounce()
                    change_show_data_ = f.change_show_data()
                    predict_ = f.predict_by_ml()
#                     f.show()
                    for k in time_band:
                        print(i, j, k)
    #                     print(pd.DataFrame(use_today_.index))
    #                     print((pd.DataFrame(use_today_.index)['TIME_BAND']==int(k)).any())
    #                     print(f.necessary_data()['TIME_BAND'])
#                         print(gateway, j, k, str(use_by_time_.loc[int(k), 'GIVE_WEIGHT_ONOFF']), str(use_today_.loc[int(k), 'ONOFF_TV']), str(bounce_rate_), alarm_)
    #                     print('INSERT INTO BOUNCE_RATE (GATEWAY_ID, COLLECT_DATE, TIME_BAND, ONOFF_AVG, ONOFF_TODAY, BOUNCE_RATE, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(j) + "'" + ', ' + str(k) + ', ' + str(f.use_by_time().loc[k, 'GIVE_WEIGHT_ONOFF']) + ', ' + str(f.use_today().loc[k, 'ONOFF_TV']) + ', ' + str(f.bounce_rate()) + ', ' + "'" + f2.alarm() + "'" + ')')
                        sql ='DELETE FROM AH_STATUS WHERE GATEWAY_ID=' + "'" + gateway + "'" + ' AND COLLECT_DATE=' + "'" + str(j) + "'" + ' AND TIME_BAND=' + "'" + k + "'"
                        cursor.execute(sql)
                        if (pd.DataFrame(use_today_.index)['TIME_BAND']==int(k)).any() & (pd.DataFrame(use_by_time_.index)['TIME_BAND']==int(k)).any():
                            sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, ONOFF_AVG, ONOFF_AVG_INT, PREDICT, ONOFF_TODAY, MEAN_BOUNCE_RATE, BOUNCE_RATE, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(j) + "'" + ', ' + "'" + k + "'" + ', ' + str(use_by_time_.loc[int(k), 'GIVE_WEIGHT_ONOFF']) + ', ' + str(change_show_data_.loc[int(k), 'AVG']) + ', ' + str(predict_.loc[int(k), 'ONOFF_TV']) + ', ' + str(change_show_data_.loc[int(k), 'TODAY']) + ', ' + str(mean_bounce_) + ', ' + str(bounce_rate_) + ', ' + str(alarm_) + ')'
                            cursor.execute(sql)
                        elif (pd.DataFrame(use_by_time_.index)['TIME_BAND']==int(k)).any() == False & (pd.DataFrame(use_today_.index)['TIME_BAND']==int(k)).any() == True:
                            sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, PREDICT, ONOFF_TODAY, MEAN_BOUNCE_RATE, BOUNCE_RATE, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(j) + "'" + ', ' + "'" + k + "'" + ', ' + str(predict_.loc[int(k), 'ONOFF_TV']) + ', ' + str(change_show_data_.loc[int(k), 'TODAY']) + ', ' + str(mean_bounce_) + ', ' + str(bounce_rate_) + ', ' + str(alarm_) + ')'
                            cursor.execute(sql)
                        elif (pd.DataFrame(use_today_.index)['TIME_BAND']==int(k)).any() == False & (pd.DataFrame(use_by_time_.index)['TIME_BAND']==int(k)).any() == True:
                            sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, ONOFF_AVG, ONOFF_AVG_INT, PREDICT, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(j) + "'" + ', ' + "'" + k + "'" + ', ' + str(use_by_time_.loc[int(k), 'GIVE_WEIGHT_ONOFF']) + ', ' + str(change_show_data_.loc[int(k), 'AVG']) + ', ' + str(predict_.loc[int(k), 'ONOFF_TV']) + ', ' + str(0) + ')'
                            cursor.execute(sql)
                        else:
                            sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(j) + "'" + ', ' + "'" + k + "'" + ', ' + str(0) + ')'
                            cursor.execute(sql)
                            
                        if (str(j) == datetime.datetime.now().strftime('%Y%m%d')) & (str(k) == str(cur)):
                            break
                else:
                    for k in time_band:
                        sql ='DELETE FROM AH_STATUS WHERE GATEWAY_ID=' + "'" + gateway + "'" + ' AND COLLECT_DATE=' + "'" + str(j) + "'" + ' AND TIME_BAND=' + "'" + k + "'"
                        cursor.execute(sql)
                        sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(j) + "'" + ', ' + "'" + k + "'" + ', ' + str(0) + ')'
                        cursor.execute(sql)
                        pass
        conn.commit()
    conn.close()            


