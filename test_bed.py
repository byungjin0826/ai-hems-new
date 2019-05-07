from utils import *

# todo: sql 문에서 이름으로 선택 가능하게...
# todo: submeter가 없는 집은 플러그 전체를 합산하여 계산하게

gateway_id = 'ep1827-tdhfmvvc-0486'

device_address = {
    '별채전체전력': '000D6F000C13E894',
    '사랑채전체전력': '000D6F000C140EF9',
    '안채전체': '000D6F000C13DC75'
}

sql = f"""
SELECT *
FROM ah_log_meter_201903
WHERE 1=1
AND DEVICE_address = '{device_address['별채전체전력']}'
"""

sql = f"""
SELECT *
FROM ah_log_meter_201903
WHERE 1=1
AND DEVICE_address = '000D6F000C13DC75'
"""

df = get_table_from_db(sql, db='aihems_service_db')

x, y = split_x_y(df, x_col='collected_date', y_col='energy')

model, params = select_regression_model('linear regression')

gs = sk.model_selection.GridSearchCV(estimator=model,
                                     param_grid=params,
                                     cv=5,
                                     n_jobs=-1)

gs.fit(x, y)


# 안채
