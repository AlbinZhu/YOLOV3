"""
@author:      Swing
@create:      2020-05-11 15:19
@desc:
"""

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a[..., 1]
print(b)