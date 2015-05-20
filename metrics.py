"""The metrics module implements functions assessing prediction error for specific purposes."""

import numpy as np

def trapz(x, y):
    """Trapezoidal rule for integrating
    the curve defined by x-y pairs.
    Assume x and y are in the range [0,1]
    """
    assert len(x) == len(y), 'x and y need to be of same length'
    x = np.concatenate([x, np.array([0.0, 1.0])])
    y = np.concatenate([y, np.array([0.0, 1.0])])
    sort_idx = np.argsort(x)
    sx = x[sort_idx]
    sy = y[sort_idx]

    area = 0.0
    for ix in range(len(x)-1):
        area += 0.5*(sx[ix+1]-sx[ix])*(sy[ix+1]+sy[ix])

    return area

def auc(y_true, y_score):

    for y in y_true:
        if(y > 1 or y < 0):
            raise ValueError("y_true contains not valid data")

    assert len(y_true) == len(y_score), "y_true and y_score must be of the same length"

    return np.trapz(y_true, x=y_score)

if __name__ == '__main__':
    print(auc([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8]))