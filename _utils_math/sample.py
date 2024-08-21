
from _utils_import import np

def sample_from_gaussian_01(shape):
    return np.random.normal(size=shape, loc=0.0, scale=1.0)
