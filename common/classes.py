from collections import namedtuple


class Device:

    def __init__(self, device_id, gateway_id):
        self.device_id = device_id
        self.gateway_id = None
        self.registered_date = None
        self.using_energy = None
        self.threshold = None

    def register(self):
        self.data = ''

    @staticmethod
    def check_list():
        return 0
