class Flight:

    def __init__(self):
        print('init')
        super().__init__()

    def __new__(cls):
        print('new')
        return super().__new__(cls)

    def number(self):
        return 'SN060'