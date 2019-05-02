from utils import *

# todo: sql 문에서 이름으로 선택 가능하게...
# todo: submeter가 없는 집은 플러그 전체를 합산하여 계산하게

sql = """
SELECT *
FROM ah_log_meter_201903
WHERE 1=1
AND DEVICE_address = '000D6F000C138617'
"""

df = get_table_from_db(sql, db = 'aihems_service_db')

x, y = split_x_y(df, x_col='collected_date', y_col='energy')

gs = sk.model_selection.GridSearchCV(estimator=regressions['linear regression'][0],
                                param_grid=regressions['linear regression'][1])

y = df.energy

gs.fit(x, y)