'''
Created on 2019. 8. 20.

@author: user
'''

######## 30분마다 데이터 갱신하는 파일 #############
from silvercare import data_handler as dh
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
    
    for i in range(0, 30):
        gateway = gateway_list.values[i][0]
        if gateway == 'ep18270158' or gateway == 'ep18270186' or gateway == 'ep18270394' or gateway == 'ep18270397' or gateway == 'ep18270408':
            pass
        else:
            print(i)
            f = dh.GetData(gateway, today)
            idx = time_band.index(cur)
            
            if (f.necessary_data()['COLLECT_DATE']==str(today)).any():
                f2 = dh.CompareBounce(gateway, today)
                f.bounce_rate_store() # 어제의 이탈율 DB에 저장 # 하루마다 갱신
                use_by_time_ = f.use_by_time()
                use_today_ = f.use_today()
                bounce_rate_ = f.bounce_rate()
                alarm_ = f2.alarm()
                mean_bounce_ = f2.mean_bounce()
                change_show_data_ = f.change_show_data()
                predict_ = f.predict_by_ml()
#                 f.show()
                
                for j in range(0, idx + 1):
                    k = time_band[j]
                    sql ='DELETE FROM AH_STATUS WHERE GATEWAY_ID=' + "'" + gateway + "'" + ' AND COLLECT_DATE=' + "'" + str(today) + "'" + ' AND TIME_BAND=' + "'" + k + "'"
                    cursor.execute(sql)
                    if (pd.DataFrame(use_today_.index)['TIME_BAND']==int(k)).any() & (pd.DataFrame(use_by_time_.index)['TIME_BAND']==int(k)).any():
                        sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, ONOFF_AVG, ONOFF_AVG_INT, PREDICT, ONOFF_TODAY, MEAN_BOUNCE_RATE, BOUNCE_RATE, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(today) + "'" + ', ' + "'" + k + "'" + ', ' + str(use_by_time_.loc[int(k), 'GIVE_WEIGHT_ONOFF']) + ', ' + str(change_show_data_.loc[int(k), 'AVG']) + ', ' + str(predict_.loc[int(k), 'ONOFF_TV']) + ', ' + str(change_show_data_.loc[int(k), 'TODAY']) + ', ' + str(mean_bounce_) + ', ' + str(bounce_rate_) + ', ' + str(alarm_) + ')'
                        cursor.execute(sql)
                    elif (pd.DataFrame(use_by_time_.index)['TIME_BAND']==int(k)).any() == False & (pd.DataFrame(use_today_.index)['TIME_BAND']==int(k)).any() == True:
                        sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, PREDICT, ONOFF_TODAY, MEAN_BOUNCE_RATE, BOUNCE_RATE, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(today) + "'" + ', ' + "'" + k + "'" + ', ' + str(predict_.loc[int(k), 'ONOFF_TV']) + ', ' + str(change_show_data_.loc[int(k), 'TODAY']) + ', ' + str(mean_bounce_) + ', ' + str(bounce_rate_) + ', ' + str(alarm_) + ')'
                        cursor.execute(sql)
                    elif (pd.DataFrame(use_today_.index)['TIME_BAND']==int(k)).any() == False & (pd.DataFrame(use_by_time_.index)['TIME_BAND']==int(k)).any() == True:
                        sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, ONOFF_AVG, ONOFF_AVG_INT, PREDICT, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(today) + "'" + ', ' + "'" + k + "'" + ', ' + str(use_by_time_.loc[int(k), 'GIVE_WEIGHT_ONOFF']) + ', ' + str(change_show_data_.loc[int(k), 'AVG']) + ', ' + str(predict_.loc[int(k), 'ONOFF_TV']) + ', ' + str(0) + ')'
                        cursor.execute(sql)
                    else:
                        sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(today) + "'" + ', ' + "'" + k + "'" + ', ' + str(0) + ')'
                        cursor.execute(sql)
            else:
                for j in range(0, idx + 1):
                    k = time_band[j]
                    sql ='DELETE FROM AH_STATUS WHERE GATEWAY_ID=' + "'" + gateway + "'" + ' AND COLLECT_DATE=' + "'" + str(today) + "'" + ' AND TIME_BAND=' + "'" + k + "'"
                    cursor.execute(sql)
                    sql ='INSERT INTO AH_STATUS (GATEWAY_ID, COLLECT_DATE, TIME_BAND, STATUS) VALUES (' + "'" + gateway + "'" + ', ' + "'" + str(today) + "'" + ', ' + "'" + k + "'" + ', ' + str(0) + ')'
                    cursor.execute(sql)
                
        conn.commit()
    conn.close()  
            
#         print(gateway)
#         print(f.use_today()) # 오늘 TV시청 data # 30분마다 갱신
#         print(f.use_24hours()) # 최근 24시간 TV시청 data # 30분마다 갱신
#         print(f.bounce_rate()) # 최근 24시간 누적 이탈율 # 30분마다 갱신
#         print(f2.bounce_list()) # 지난 6일간의 이탈율 list #DB에서 꺼내옴
#         f2.alarm() # 알람 # 30분마다 갱신