
import sys
import pandas as pd
import numpy as np
import manager


class SOLTCalibration:
    """ 
    One-port: 3-Term Error Model

    s0 o--->---+------->------+--->---+
               |      e10     |       |
               v e00      e11 ^       v G
               |      e01     |       |
    b0 o---<---+-------<------+---<---+

    Two-port: 6-Term Error Model

             +----------------------->-----------------------+
             |                      e30                      |
    a0 o-->--+----->-----+-->--+----->-----+-->--+----->-----+-->--o b3
             |     1     |     |    S21    |     |  e10 e32
             v e00   e11 ^     v S11   S22 ^     v e22
             |  e10 e01  |     |    S12    |     |
    b0 o--<--+-----<-----+--<--+-----<-----+--<--+

    (The 12 term model includes more leakage terms: e20, e02, e31, e21, e12)

    Two-port: 8-Term Error Model (One can be normalized to give 7 terms)

    a0 o-->--+----->-----+-->--+----->-----+-->--+----->-----+-->--o b3
             |    e10    |     |    S21    |     |    e32    |
             v e00   e11 ^     v S11   S22 ^     v e22   e33 ^
             |    e01    |     |    S12    |     |    e23    |
    b0 o--<--+-----<-----+--<--+-----<-----+--<--+-----<-----+--<--o a3
    """ 

    def __init__(self, index):
        columns = ['S11M', 'S21M', 'load', 'open', 'short', 'thru', 'thrumatch', 'crosstalk']
        data = [(0, 0, 0, 1, -1, 0, 0, 0)] 
        self.index = index
        self.forward = pd.DataFrame(data, columns=columns, index=index)
        self.reverse = pd.DataFrame(data, columns=columns, index=index)
        self.calibrate() 
        self.calibrate(reverse=True) 


    # calculate error terms

    def calibrate(self, reverse=False):
        df = self.reverse if reverse else self.forward
        gmo = df['open']
        gms = df['short'] 
        df['e00'] = df['load']
        df['e11'] = (gmo + gms - 2 * df['e00']) / (gmo - gms)
        df['e10e01'] = -2 * (gmo - df['e00']) * (gms - df['e00']) / (gmo - gms)
        df['de'] = df['e00'] * df['e11'] - df['e10e01']
        df['e30'] = df['crosstalk']
        df['e22'] = (df['thrumatch'] - df['e00']) / (df['thrumatch'] * df['e11'] - df['de'])
        df['e10e32'] = (df['thru'] - df['e30']) * (1 - df['e11'] * df['e22'])

    def update(self, gm, name, reverse=False):
        df = self.reverse if reverse else self.forward
        gm.name = name
        df[name] = gm
        self.calibrate(reverse=reverse)
        return gm


    # real-time

    def return_loss(self, gm, reverse=False):
        df = self.reverse if reverse else self.forward
        gamma = (gm - df['e00']) / (gm * df['e11'] - df['de'])
        return gamma.rename('S22' if reverse else 'S11')

    def response(self, gm, reverse=False):
        df = self.reverse if reverse else self.forward
        gamma = gm / df['e10e32']
        return gamma.rename('S12' if reverse else 'S21')

    def enhanced_response(self, gm, reverse=False):
        df = self.reverse if reverse else self.forward
        gamma = gm / df['e10e32'] * (df['e10e01'] / (df['e11'] * df['S11M'] - df['de']))
        return gamma.rename('S12' if reverse else 'S21')
        

    # batch

    def two_port(self):
        forward = self.forward
        reverse = self.reverse

        S11M, S21M = forward['S11M'], forward['S21M']
        S22M, S12M = reverse['S11M'], reverse['S21M']
        e00, e11,  e22 = forward['e00'], forward['e11'], forward['e22']
        e33r, e22r, e11r = reverse['e00'], reverse['e11'], reverse['e22']
        e30, e10e01, e10e32 = forward['e30'], forward['e10e01'], forward['e10e32']
        e03r, e23re32r, e23re01r = reverse['e30'], reverse['e10e01'], reverse['e10e32']

        N11 = (S11M - e00) / e10e01
        N21 = (S21M - e30) / e10e32
        N22 = (S22M - e33r) / e23re32r
        N12 = (S12M - e03r) / e23re01r
        D = (1 + N11 * e11) * (1 + N22 * e22r) - N21 * N12 * e22 * e11r

        df = pd.DataFrame(index=self.index)
        df.name = 'Two Port Parameters'
        df['S11'] = (N11 * (1 + N22 * e22r) - e22 * N21 * N12) / D
        df['S21'] = (N21 * (1 + N22 * (e22r - e22))) / D
        df['S22'] = (N22 * (1 + N11 * e11) - e11r * N21 * N12) / D
        df['S12'] = (N12 * (1 + N11 * (e11 - e11r))) / D
        return df
 

# pyvna
######################################

# initialize driver

def init(**args):
    manager.init(**args)
    return version()

def version():
    return manager.driver.version()

def temperature():
    return manager.driver.temperature()

def reset():
    manager.driver.reset()

def close():
    manager.close()


# instantiate calibration object
        
def create(start=None, stop=None, points=None, calibration=SOLTCalibration):
    driver = manager.driver
    start = driver.min_freq if start is None else start
    stop = driver.max_freq if stop is None else stop
    points = driver.default_points if points is None else points
    step = (stop - start) // points

    if start > stop or stop > driver.max_freq or start < driver.min_freq: 
        raise ValueError("bad frequency range")
    if points > driver.max_points or points < 1:
        raise ValueError("bad number of points")
    if step < 1:
        raise ValueError("step size too small")
    index = np.array([int(start + i * step) for i in range(points + 1)])
    return calibration(index)


## measurements

def transmission(cal, name, average=3, window=3, reverse=False):
    gm = pd.Series(0, cal.index)
    for i in range(average):
        gm += manager.driver.transmission(cal.index, reverse=reverse)
    gm = gm / average
    gm = rolling_mean(gm, window=window)
    gm = cal.update(gm, name, reverse=reverse)
    return gm

def reflection(cal, name, average=3, window=3, reverse=False):
    gm = pd.Series(0, cal.index)
    for i in range(average):
        gm += manager.driver.reflection(cal.index, reverse=reverse)
    gm = gm / average
    gm = rolling_mean(gm, window=window)
    gm = cal.update(gm, name, reverse=reverse)
    return gm


## calibrate

def cal_open(cal, **kw):
    return reflection(cal, 'open', **kw)

def cal_short(cal, **kw):
    return reflection(cal, 'short', **kw)

def cal_load(cal, **kw):
    return reflection(cal, 'load', **kw)

def cal_thru(cal, **kw):
    return transmission(cal, 'thru', **kw)

def cal_thrumatch(cal, **kw):
    return transmission(cal, 'thrumatch', **kw)

def cal_crosstalk(cal, **kw):
    return transmission(cal, 'crosstalk', **kw)


# measure

def S11M(cal, **kw):
    return reflection(cal, 'S11M', reverse=False, **kw)

def S21M(cal, **kw):
    return transmission(cal, 'S21M', reverse=False, **kw)

def S22M(cal, **kw):
    return reflection(cal, 'S11M', reverse=True, **kw).rename('S22M')

def S12M(cal, **kw):
    return transmission(cal, 'S21M', reverse=True, **kw).rename('S12M')


# real-time corrected measurements

def return_loss(cal, **kw):
    gm = reflection(cal, 'S11M', **kw)
    return cal.return_loss(gm, reverse=kw.get('reverse'))

def response(cal, **kw):
    gm = transmission(cal, 'S21M', **kw)
    return cal.response(gm, reverse=kw.get('reverse'))

def enhanced_response(cal, **kw):
    gm = transmission(cal, 'S21M', **kw)
    return cal.enhanced_response(gm, reverse=kw.get('reverse'))

def forward_path(cal, **kw):
    gloss = return_loss(cal, **kw)
    gresp = enhanced_response(cal, **kw)
    df = pd.DataFrame(index=cal.index)
    df.name = 'Forward Parameters'
    df[gloss.name] = gloss
    df[gresp.name] = gresp
    return df


# batch corrected measurements

def two_port(cal):
    return cal.two_port()


# utilities

def rolling_mean(gm, window=3):
    re = gm.apply(np.real).rolling(window, min_periods=1).mean()
    im = gm.apply(np.imag).rolling(window, min_periods=1).mean()
    return re + 1j * im

##

def to_impedance(gm):
    z = 50 * (1 + gm) / (1 - gm)
    return z.rename(gm.name + ' (impedance)')

def to_admittance(gm):
    z = 50 * (1 + gm) / (1 - gm)
    y = 1 / z
    return y.rename(gm.name + ' (admittance)')

def to_parallel(gm):
    z = 50 * (1 + gm) / (1 - gm)
    y = 1 / z
    xp = pd.Series(1 / y.real - 1j / y.imag, gm.index)
    return xp.rename(gm.name + ' (parallel)')

def to_quality_factor(gm):
    z = 50 * (1 + gm) / (1 - gm)
    q = pd.Series(np.abs(z.imag) / z.real, gm.index)
    return q.rename(gm.name + ' (Q)')

##

def to_inductance(x):
    f = x.index.values
    im = x.imag / (2 * np.pi * f)
    # im = np.maximum(0, im)
    s = pd.Series(x.real + 1j * im, x.index, name=x.name)
    return s.rename(x.name + ' (inductance)')
    
def to_capacitance(x):
    f = x.index.values
    im = -1 / (2 * np.pi * f * x.imag)
    # im = np.maximum(0, im)
    s = pd.Series(x.real + 1j * im, x.index, name=x.name)
    return s.rename(x.name + ' (capacitance)')


# plot

def polar(gm, figsize=(6, 6), color='r', rmax=1.0, lw=2):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(polar=True)
    ax.set_title(gm.name)
    ax.plot(np.angle(gm), np.abs(gm), color, lw=lw)
    if rmax: ax.set_rmax(rmax)
    fig.tight_layout()
    plt.show()

def bode(gm, figsize=(10, 6), color=['r', 'b'], ylim=[], lw=2):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(gm.name)
    ax.set_xlabel('frequency')
    ax.plot(gm.index, 10 * np.log10(np.abs(gm)), color[0], lw=lw)
    ax.set_ylabel('gain (db)', color=color[0])
    ax.tick_params('y', colors=color[0])
    if ylim: ax.set_ylim(ylim[0])
    ax = ax.twinx()
    ax.plot(gm.index, np.degrees(np.angle(gm)), color[1], lw=lw)
    ax.set_ylabel('phase (deg)', color=color[1])
    ax.tick_params('y', colors=color[1])
    if ylim: ax.set_ylim(ylim[1])
    fig.tight_layout()
    plt.show()


def plot(gm, figsize=(10, 6), color=['r', 'b'], ylim=[], lw=2):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(gm.name)
    ax.set_xlabel('frequency')
    ax.plot(gm.index, np.real(gm), color[0], lw=lw)
    ax.set_ylabel('real', color=color[0])
    ax.tick_params('y', colors=color[0])
    if ylim: ax.set_ylim(ylim[0])
    ax = ax.twinx()
    ax.plot(gm.index, np.imag(gm), color[1], lw=lw)
    ax.set_ylabel('imag', color=color[1])
    ax.tick_params('y', colors=color[1])
    if ylim: ax.set_ylim(ylim[1])
    fig.tight_layout()
    plt.show()


def cartesian(gm, figsize=(10, 6), color='r', ylim=None, lw=2):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(gm.name)
    ax.plot(np.real(gm), np.imag(gm), color, lw=lw)
    ax.set_xlabel('real')
    ax.set_ylabel('imag', color=color)
    if ylim: ax.set_ylim(ylim)
    fig.tight_layout()
    plt.show()




"""
def two_port_hp(cal):
    return cal.two_port_hp()

    def two_port_hp(self):
        forward = self.forward
        reverse = self.reverse

        S11M = forward['S11M']
        S21M = forward['S21M']
        e00 = forward['e00']
        e11 = forward['e11']
        e22 = forward['e22']
        e30 = forward['e30']
        e10e01 = forward['e10e01']
        e10e32 = forward['e10e32']

        S22M = reverse['S11M']
        S12M = reverse['S21M']
        e33r = reverse['e00']
        e22r = reverse['e11']
        e11r = reverse['e22']
        e03r = reverse['e30'] 
        e23re32r = reverse['e10e01'] 
        e23re01r = reverse['e10e32']

        ds = (
            (1 + (S11M - e00) / e10e01 * e11) *
            (1 + (S22M - e33r) / e23re32r * e22r) -
            (S21M - e30) / e10e32 *
            (S12M - e03r) / e23re01r *
            e22 * e11r)

        df = pd.DataFrame(index=self.index)
        df.name = 'Two Port Parameters'
        df['S11'] = (
            (S11M - e00) / e10e01 *
            (1 + (S22M - e33r) / e23re32r * e22r) -
            e22 *
            (S21M - e30) / e10e32 *
            (S12M - e03r) / e23re01r
        ) / ds
        df['S22'] = (
            (S22M - e33r) / e23re32r  *
            (1 + (S11M - e00) / e10e01 * e11) -
            e11r *
            (S21M - e30) / e10e32 *
            (S12M - e03r) / e23re01r
        ) / ds
        df['S21'] = (
            (S21M - e30) / e10e32 *
            (1 + (S22M - e33r) / e23re32r * 
            (e22r - e22))
        ) / ds
        df['S12'] = (
            (S12M - e03r) / e23re01r *
            (1 + (S11M - e00) / e10e01 * 
            (e11 - e11r))
        ) / ds
        return df
"""

