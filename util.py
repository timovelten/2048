import numpy as np
from io import BytesIO

def save_ndarray(arr: np.ndarray):
    out = BytesIO()
    np.save(out, arr)
    return out.getvalue()

def load_ndarray(bs: bytes):
    return np.load(BytesIO(bs))

def softmax(x, axis=None):
    # Thanks to https://stackoverflow.com/a/50425683/3398054 
    # (this is essentially how it is done in the scipy implementation)
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)