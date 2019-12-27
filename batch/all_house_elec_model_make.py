import requests
import common.data_load as dl
import common.model_training as mt


if __name__ == '__main__':
    house_list = dl.device_info().HOUSE_NO.unique()
    for house_no in house_list:
        try:
            mt.make_model_elec(house_no=house_no)
        except:
            print('error')
