import datetime
import pandas as pd
import settings

# def datetime_decorator(func):
#     def decorated():
#         print(datetime.datetime.now())
#         func()
#         print(datetime.datetime.now())
#     return decorated()
#
#
# # 먼저 decorator 역할을 하는 함수를 정의하고, 이 함수에서 decorator 가 적용될 함수를 인자로 받는다.
# # python 은 함수의 인자로 다른 함수를 받을 수 있다는 특징을 이용하는 것이다.
# # decorator 역할을 하는 함수 내부에 또 한번 함수를 선언(nested function)하여 여기에 추가적인 작업(시간 출력)을 선언해 주는 것이다.
# # nested 함수를 return 해주면 된다.
#
# @datetime_decorator
# def main_function_1():
#     print("Main Function 1 Start")
#
#
# if __name__ == '__main__':
#     main_function_1()
#
# class DatetimeDecorator:
#     def __init__(self, f):
#         self.func = f
#
#     def __call__(self, *args, **kwargs):
#         print(datetime.datetime.now())
#         self.func(*args, **kwargs)
#         print(datetime.datetime.now())
#
#
# class MainClass:
#     def __init__(self):
#         self.func = f
#         print("")
#
#     @DatetimeDecorator
#     def main_function_1(self):
#         print("main 1")
#
#     def main_function_2(self):
#         print("main 2")
#
#     @DatetimeDecorator
#     def main_function_3(self):
#         print("main 3")
#
#
# if __name__ == '__main':
#     my = MainClass()
#     my.main_function_1()
#     my.main_function_2()
#     my.main_function_3()
#
# from functools import wraps
#
#
# def my_decorator(func):
#     @wraps(func)
#     def runs_func():
#         print("This is ")
#         func()
#         print("blog")
#     return runs_func
#
#
# @my_decorator
# def my_func():
#     print("yaboong's ")
#
#
# my_func()
#
#
from functools import wraps

AUTH_INFO = 'd'
DB = {}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if AUTH_INFO == 'admin':
            f(*args, **kwargs)
            print('update success')
        else:
            print('no permission')
    return decorated


@login_required
def update_data(data1, data2):
    DB['data1'] = data1
    DB['data2'] = data2


update_data(1, 2)
print(DB)


def pandas_read_sql(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return pd.read_sql(f(*args, **kwargs), con=settings.conn)
    return decorated


@pandas_read_sql
def sql_load_test():
    sql = f"""SELECT * FROM AH_DEVICE"""
    return sql


df = sql_load_test()
