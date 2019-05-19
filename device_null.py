
import pandas as pd
import numpy as np

min_freq = 1
max_freq = int(3e9)
default_points = 1000


def gamma(index):
    re = .75 * np.sin(index / index[-1] * 4 * np.pi)
    im = .75 * np.cos(index / index[-1] * 4 * np.pi)
    re += .01 * np.random.randn(len(index))
    im += .01 * np.random.randn(len(index))
    return pd.Series(re + 1j * im, index=index)


class Driver():
    max_freq = max_freq
    min_freq = min_freq
    default_points = default_points

    def __init__(self, **options):
        pass

    def close(self):
        pass

    def reset(self):
        pass

    def version(self):
        return 'null device'

    def temperature(self):
        return 25.0

    def reflection(self, index, reverse=None):
        return gamma(index)

    def transmission(self, df, reverse=None):
        return gamma(index)


