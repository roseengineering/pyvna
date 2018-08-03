
import sys
import struct
from serial import Serial
import pandas as pd
import numpy as np

min_freq = 1
max_freq = int(30e6)
max_points = 100000
default_points = 200

default_baudrate = 38400
default_port = '/dev/ttyUSB0'


def progress(count, total, transmit):
    bar_len = 60
    filled_len = int(bar_len * count / total)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    status = "transmission" if transmit else "reflection" 
    sys.stdout.write('[%s] %s / %s points (%s)\r' % (bar, count, total, status))
    sys.stdout.flush()
    if count >= total: sys.stdout.write('\n')

def writeln(fd, line):
    fd.write(line.encode())
    fd.write(b'\r\n')

def readln(fd):
    return fd.readline().decode().strip()

def read_sample(fd, freq):
    line = readln(fd)
    d = line.split(',')
    f = round(float(d[0]) * 1e6)
    if freq != f: print('WARN', freq, f, freq - f)
    re = float(d[1])
    im = float(d[2])
    return np.complex(re, im)
    
def gamma(fd, index, transmit=False):
    start = index[0]
    stop = index[-1]
    points = len(index)
    zo = np.complex(50)

    # start scan
    writeln(fd, "fq%d" % ((start + stop) // 2))
    readln(fd)  # 'OK'
    writeln(fd, "sw%d" % (stop - start))
    readln(fd)  # 'OK'
    writeln(fd, "frx%d" % (points - 1))

    # read scan
    data = []
    for i in range(points):
        progress(i, points - 1, transmit)
        z = read_sample(fd, index[i])
        data.append((z - zo) / (z + zo))

    # end scan
    line = readln(fd)  # 'OK'
    return pd.Series(data, index=index)

 
class Driver():
    max_freq = max_freq
    min_freq = min_freq
    max_points = max_points
    default_points = default_points

    def __init__(self, **options):
        port = options.get('port', default_port)
        baudrate = int(options.get('baudrate', default_baudrate))
        self.fd = Serial(port, baudrate)

    def reset(self):
        fd = self.fd
        fd.close()
        fd.open()

    def close(self):
        self.fd.close()

    def version(self):
        fd = self.fd
        writeln(fd, "ver")
        return readln(fd)

    def temperature(self):
        pass

    def reflection(self, index, reverse=None):
        return gamma(self.fd, index)

    def transmission(self, index, reverse=None):
        pass

    ###

    def sweep(self, index, reverse=None):
        gamma(self.fd, index)

