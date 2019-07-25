import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from datetime import date

date1 = pd.Series(pd.date_range('2012-1-1 12:00:00', periods=7, freq='M'))
date2 = pd.Series(pd.date_range('2013-3-11 21:45:00', periods=7, freq='W'))

df = pd.DataFrame(dict(Start_date=date1, End_date=date2))
print(df)