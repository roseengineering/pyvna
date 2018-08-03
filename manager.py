
import sys
from configparser import ConfigParser


class FakeDriver:
    def close(self):
        pass

    def __getattr__(self, name):
        if name[0] == '_': return
        message = "Please initialize the device"
        print(message, file=sys.stderr)
        raise ValueError(message)


class Manager:
    def __init__(self):
        self.driver = FakeDriver()

    def init(self, **args):
        config = ConfigParser()
        config.read('device.conf')
        options = dict(config['device']) if 'device' in config else {}

        name = args['name'] if 'name' in args else options.get('name', 'null')
        if name in config: 
            options.update(dict(config[name]))
        options.update(args)

        if type(self.driver) is FakeDriver:
            device = __import__('device_' + name)
            self.driver = device.Driver(**options)

    def close(self):
        self.driver.close()
        self.driver = FakeDriver()


sys.modules[__name__] = Manager()

