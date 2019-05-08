from utils import *

class Home:
    def __init__(self, name):
        self.name = name
        self.gateway_id = get_gateway_id(self.name)
        self.device_list = get_device_list(self.gateway_id)
        self.check_meter = check_meter(self.device_list)
        self.approval_using_auto_control = True
        self.approval_real_time_monitoring = True
        self.approval_adr = True

    def auto_control(self):
        return 0

    def real_time_monitoring(self):
        return 0

    def adr(self):
        return 0