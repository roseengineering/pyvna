
import sys
import struct
from serial import Serial
import pandas as pd
import numpy as np

min_freq = int(100e3)
max_freq = int(200e6)
default_points = 2000

prescaler = 1000000 / 8259552
default_baudrate = 115200
default_port = '/dev/ttyUSB0'


def progress(count, total, transmit):
    bar_len = 60
    filled_len = int(bar_len * count / total)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    status = "transmission" if transmit else "reflection" 
    sys.stdout.write('[%s] %s / %s points (%s)\r' % (bar, count, total, status))
    sys.stdout.flush()
    if count >= total: sys.stdout.write('\n')

def writeln(fd, buf=''):
    fd.write(str(buf).encode('latin-1'))
    fd.write(b'\r')

def read_short(fd):
    r = fd.read(2)
    return struct.unpack('<H', r)[0]

def readln(fd):
    line = b''
    while True:
        ch = fd.read(1)
        if ch == b'\n': break
        line += ch
    return line[:-1].decode('latin-1')

def read_sample(fd):
    x1 = read_short(fd)
    x2 = read_short(fd)
    x3 = read_short(fd)
    x4 = read_short(fd)
    re = x1 - x3
    im = x2 - x4
    return np.complex(re, im)

def gamma(fd, index, transmit=False):
    start = index[0]
    stop = index[-1]
    points = len(index)
    steps = (stop - start) / points
    max_value = 2**16

    # start scan
    writeln(fd, 0 if transmit else 1)
    writeln(fd, int(start / prescaler))
    writeln(fd, 0)
    writeln(fd, points)
    writeln(fd, int(steps / prescaler))

    # read scan
    data = []
    for i in range(points):
        progress(i, points - 1, transmit)
        c = read_sample(fd)
        data.append(c / max_value)

    # return dataframe
    return pd.Series(data, index=index, dtype=np.complex)

 
class Driver():
    max_freq = max_freq
    min_freq = min_freq
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

    def voltage(self):
        fd = self.fd
        writeln(fd, 8)
        return read_short(fd) * 6 / 1024

    def version(self):
        fd = self.fd
        writeln(fd, 9)
        return readln(fd)

    def temperature(self):
        pass

    def reflection(self, index, reverse=None):
        return gamma(self.fd, index)

    def transmission(self, index, reverse=None):
        return gamma(self.fd, index, transmit=True)

    ###

    def sweep(self, index, reverse=None):
        gamma(self.fd, index)


