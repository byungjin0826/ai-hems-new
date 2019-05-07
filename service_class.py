from utils import *

class Home:
    import utils
    def __init__(self):
        self.name = input()
        self.gateway_id = get_gateway_id(self.name)
        self.device_list = get_device_list(self.gateway_id)