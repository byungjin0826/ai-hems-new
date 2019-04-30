# member_name = input('사용자 이름: ')
# appliance_name = input('가전기기 이름: ')
# # start = input('시작일: ')
# end = input('종료일: ')
#
# start = time.time()
# gs, X, Y = make_prediction_model(member_name=member_name, appliance_name=appliance_name)
# end = time.time()
# print('걸리시간: ', round(end-start, 3), 's', sep = "")
# print('정확도: ', round(gs.best_score_, 3) , sep = "")
# #
# # model_fitted = gs
# # model_loaded = load('./')
# #
# # model_fitted.predict()
#
# df = read_db_table(member_name= member_name, appliance_name = appliance_name,  start = '2019-03', end = '2019-04')
#
# df1 = set_data(df)
# X, Y = split_x_y(df1)
#
# Y = gs.predict(X)
#
# df1.appliance_status = Y
# gateway_id = df1.gateway_id[0]
# df1.gateway_id = gateway_id[:6] + gateway_id[-4:]
#
# df2 = transform_data(df1)
#
# # write_db(df2)


gs = make_prediction_model(member_name='박재훈', appliance_name='TV')

df = read_db_table(member_name='사랑채', appliance_name='사랑채TV', start='2019-03', end = '2019-04')

df['energy_diff'] = df.energy - df.energy.shift(1)



# df = transform_data(df)
# write_db(df)