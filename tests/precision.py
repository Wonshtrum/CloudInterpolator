from cloud_interpolator import cloud2cloud
import numpy as np
from time import time

SIZE_TEST= 30
NVAR = 100
K = 4

xyz_source = np.random.rand(SIZE_TEST*1, SIZE_TEST*2, SIZE_TEST*3, 3)
xyz_target = np.random.rand(SIZE_TEST*4, SIZE_TEST*5, 3)

rad_source = np.linalg.norm(xyz_source,axis=-1)
rad_tgt = np.linalg.norm(xyz_target,axis=-1)

raw_data_array = np.repeat(np.expand_dims(rad_source, axis=-1),NVAR, axis=-1)

function = lambda dists: np.power(dists, 4)

t = time()
out = cloud2cloud(xyz_source, raw_data_array, xyz_target, function=function, stencil=K, verbose=True)
print(time()-t)

print(out.shape)

max_error = np.max(np.abs(out[:,:,0]-rad_tgt))
print("Error vs ref:", max_error)
