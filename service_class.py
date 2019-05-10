from utils import *
import datetime

class Home:
    def __init__(self, name):
        self.name = name
        self.gateway_id = get_gateway_id(self.name)
        self.device_list = get_device_list(self.gateway_id)
        self.check_meter = check_meter(self.device_list)
        self.approval_using_auto_control = False
        self.approval_real_time_monitoring = False
        self.approval_adr = False

    def auto_control(self):
        for device_id in self.device_list:
            self.schedule = calc_weekly_schedule(device_id) # data_frame 형태로 나옴.., 날짜, 시간, 변화하고자 하는 상태

        return 0

    def real_time_monitoring(self):
        return 0

    def adr(self):
        return 0


class Appliance:
    def __init__(self):
        self.name = None
        self.using_elec_per_hour = None
        self.standby_elec_per_hour = None
        self.connected_date = None
        self.appliance_no = None

class Device:
    def __init__(self):
        self.device_id = None
        self.device_type = None

class Gateway:
    def __init__(self):
        self.gateway_id = None
        self.house_name_installed = None