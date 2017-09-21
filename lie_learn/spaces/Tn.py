"""
The n-Torus
"""

import numpy as np

def linspace(b, n=1, convention='regular'):
    if convention == 'regular':
        res = []
        for i in range(n):
            res.append(np.arange(b) * 2 * np.pi / b)

    else:
        raise ValueError('Unknown convention:' + convention)

    return res