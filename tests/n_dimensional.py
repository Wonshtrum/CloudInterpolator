from cloud_interpolator import CloudInterpolator
import numpy as np
from time import time


points_src = 100000
points_tgt = 1000
x = np.random.rand(3, points_src)
xi = np.random.rand(3, points_tgt)

data = np.random.rand(points_src, 40, 50, 3)

print("start")
t = time()
base = CloudInterpolator(x, xi)
print("interp")
result = base.interp(data)
print(time()-t)

print(result.shape)
