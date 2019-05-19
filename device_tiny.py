
import sys
import struct
from serial import Serial
import pandas as pd
import numpy as np

min_freq = int(1e6)
max_freq = int(3e9)
default_points = 2000

prescaler = 10
switch_points = [ 1.045e9, 1.525e9 ]
default_baudrate = 921600
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

def read_half(fd):
    r = fd.read(3)
    return struct.unpack('<i', r + b'\0')[0]

def readln(fd):
    line = b''
    while True:
        ch = fd.read(1)
        if ch == b'\n': break
        line += ch
    return line[:-1].decode('latin-1')

def read_sample(fd):
    x1 = read_half(fd)
    x2 = read_half(fd)
    x3 = read_half(fd)
    x4 = read_half(fd)
    re = x1 - x3
    im = x2 - x4
    return np.complex(re, im)

def gamma(fd, index, transmit=False):
    start = index[0]
    stop = index[-1]
    points = len(index)
    max_value = 2 ** 24

    # start scan
    writeln(fd, 6 if transmit else 7)
    writeln(fd, int(start / prescaler))
    writeln(fd, int(stop / prescaler))
    writeln(fd, points)
    writeln(fd)

    # read scan
    data = []
    for i in range(points):
        progress(i, points - 1, transmit)
        c = read_sample(fd)
        data.append(c / max_value)

    # end scan
    writeln(fd, 7)
    writeln(fd, 0)
    writeln(fd, 0)
    writeln(fd, 1)
    writeln(fd, 0)
    read_sample(fd)

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

    def version(self):
        fd = self.fd
        writeln(fd, 9)
        return readln(fd)

    def temperature(self):
        fd = self.fd
        writeln(fd, 10)
        return read_short(fd) / 10

    def reflection(self, index, reverse=None):
        return gamma(self.fd, index)

    def transmission(self, index, reverse=None):
        return gamma(self.fd, index, transmit=True)

    ###

    def sweep(self, index, reverse=None):
        gamma(self.fd, index)


