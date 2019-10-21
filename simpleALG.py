import pandas as pd
import time

if __name__ == '__main__':
    print('----------Start------------')

    timeArr = [(str(x//60) if len(str(x//60))==2 else ("0"+str(x//60)) )+
               (str(x % 60) if len(str(x % 60))==2 else ("0"+str(x % 60))) for x in range(1,1441)]

    tt = pd.read_csv(f'duration/20190629_update5_duration60.txt',
                      names=['HOUSE_NO', 'COLLECT_DATE', 'COLLECT_TIME', 'AVG', 'CRT_AVG'],
                      dtype={'COLLECT_TIME': str},
                      index_col=False)

    timeArr5 = tt['COLLECT_TIME'].values

    today = 20190629



    # 1분 간격 / 30일 동안 65%이상 1 이면 1
    Data = pd.read_csv(f"show__11.txt",
                        names=['COLLECT_TIME', 'CRT', 'AVG', 'AVG65'],
                        dtype={'COLLECT_TIME': str, 'ONOFF': str},
                        index_col=False)



    ta = 0
    danger = 0
    specialMoment = 0
    status = "양호"
    avgWaching = 120
    f = open('make/result_avg.txt','w',encoding='utf-8')

    for num, t in enumerate(timeArr) :
        if(num != 0):
            pre_row = Data.loc[num-1]
        else:
            pre_row = Data.loc[num]
        row = Data.loc[num]

        # 평균 패턴과 다른 경우
        if not row.AVG==row.CRT:
            # 평소엔 안보는데 현재 보고있는 경우
            if (str(row.CRT) == '1') and (str(row.AVG) == '0'):
                specialMoment += 1    # 그날 특별하게 보는것일수 있기때문에 천천이 카운트
                # print(f"[{t}] 평소엔 안보는데 현재 보고있는 경우")
            # 평소엔 보는데 현재 안보고 있는 경우
            elif (str(row.CRT) == '0') and (str(row.AVG) == '1'):
                danger += 1
                # print(f"[{t}] 평소엔 보는데 현재 안보고 있는 경우")

        # 평균 패턴과 같은 경우
        elif (row.AVG == row.CRT):
            if pre_row.CRT == row.CRT:
                pass
            # 평균 패턴과 같은데 0인경우
            elif str(row.CRT) == '0':
                danger = 0
                specialMoment = 0
                status = "양호"
                # print(f"[{t}] 평균 패턴과 같은데 0인경우")
            # 평균 패턴과 같은데 1인경우
            elif str(row.CRT) == '1':
                danger = 0
                specialMoment = 0
                status = "양호"
                # print(f"[{t}] 평균 패턴과 같은데 1인경우")
        else:
            pass

        if   (avgWaching <= danger and danger < avgWaching*2) \
           or(avgWaching*1.5 <= specialMoment and specialMoment <avgWaching*2.5):
            status = "주의"
        elif (avgWaching*2 <= danger) or (avgWaching*2.5 <= specialMoment):
            status = "방문요망"

        # 현재상태가 변화가 있으면
        if pre_row.CRT != row.CRT:
            status = "양호"
            danger = 0
            specialMoment = 0


        if timeArr[num] == timeArr5[ta]:
            print(f'{row.COLLECT_TIME},{row.AVG},{row.CRT},{status}')
            data = f'{row.COLLECT_TIME},{row.AVG},{row.CRT},{status}'+'\n'
            f.write(data)
            ta += 1
            if (ta == 288):
                f.close()
                break

    Data = pd.read_csv(f"make/result_avg.txt",
                       names=['COLLECT_TIME', 'AVG', 'CRT', 'STATS'],
                       dtype={'COLLECT_TIME': str},
                       index_col=False,
                       encoding = 'utf-8-sig')

    Data.to_csv('make/BBresult.csv', header=False, index=False, encoding='utf-8-sig')