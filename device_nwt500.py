
import struct
from serial import Serial
import pandas as pd
import numpy as np

min_freq = int(100e3)
max_freq = int(500e6)
max_points = 100000
default_points = 1000

default_baudrate = 57600
default_port = '/dev/ttyUSB0'

def progress(count, total, transmit):
    bar_len = 60
    filled_len = int(bar_len * count / total)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    status = "transmission" if transmit else "reflection" 
    sys.stdout.write('[%s] %s / %s points (%s)\r' % (bar, count, total, status))
    sys.stdout.flush()
    if count >= total: sys.stdout.write('\n')

##

def read_byte(fd):
    r = fd.read(1)
    return r[0]

def read_short(fd):
    r = fd.read(2)
    return struct.unpack('<H', r)[0]

def read_sample(fd):
    c1 = read_short(fd)
    c2 = read_short(fd)
    return c1

def send_command(fd, line):
    fd.write(b'\x8f')
    fd.write(line.encode('latin-1'))

def convert_to_dbm(v):
    mfactor = 0.193143
    mshift = -84.634597
    return mfactor * v + mshift

##

def gamma(fd, index, transmit=False):
    start = index[0]
    stop = index[-1]
    points = len(index)
    data = []
    send_command(fd, "x%09d%08d%04d" % (start, step, points))
    for i in range(points):
        progress(i, points - 1, transmit)
        c = read_sample(fd)
        db = convert(c)
        mw = 10 ** (db / 10)
        data.append(mw)
    return pd.Series(data, index=index, dtype=np.complex)


class Driver():
    max_freq = max_freq
    min_freq = min_freq
    max_points = max_points
    default_points = default_points

    def __init__(self, **options):
        port = options.get('port', default_port)
        baudrate = int(options.get('baudrate', default_baudrate))
        self.fd = Serial(port, baudrate)

    def close(self):
        self.fd.close()

    def reset(self):
        fd = self.fd
        fd.close()
        fd.open()

    def version(self):
        fd = self.fd
        send_command(fd, "v")
        r = read_byte(fd)
        return "%.2f" % (r / 100)

    def temperature(self):
        pass

    def reflection(self, index, reverse=None):
        return gamma(self.fd, index)

    def transmission(self, index, reverse=None):
        return gamma(self.fd, index, transmit=True)

    ##

    def sweep(self, index, reverse=None):
        gamma(self.fd, index)

    def set_frequency(self, freq):
        send_command(self.fd, "f%09d" % int(freq))

    def set_attenuation(self, db=None):
        db = 0 if db is None else db
        lookup = { 0:0, 10:1, 20:2, 30:3, 40:6, 50:7 }
        value = lookup[db]
        send_command(self.fd, "r%c" % value)

    def read_power(self, count=None):
        count = 5 if count is None else count
        fd = self.fd
        total = 0
        for n in range(count):
            send_command(fd, "m")
            total += read_sample(fd)
        return convert_to_dbm(total / count)

