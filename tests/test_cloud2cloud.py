from cloud2cloud import CloudInterpolator, cloud2cloud
import numpy as np


def squared_func(xxx, yyy, zzz):
	return (3 * xxx ** 2 + 5 * yyy ** 2 + 7 * zzz * 2) / (
		3 ** 2 * 5 ** 2 * 7 ** 2)


def test_cloud2cloud():
	DIM1 = 5
	DIM2 = 7
	size = 1000

	x_coor = np.linspace(0, 1., DIM2 * 10)
	y_coor = np.linspace(0, 1., DIM2 * 10)
	z_coor = np.linspace(0, 1., DIM2 * 10)

	source_xyz = np.stack(np.meshgrid(x_coor, y_coor, z_coor),
		axis=-1)
	target_xyz = np.stack((np.random.rand(DIM1 * size) * 1.,
		np.random.rand(DIM1 * size) * 1.,
		np.random.rand(DIM1 * size) * 1.),
		axis=-1)

	source_val = squared_func(np.take(source_xyz, 0, axis=-1),
		np.take(source_xyz, 1, axis=-1),
		np.take(source_xyz, 2, axis=-1))

	target_ref = squared_func(target_xyz[:, 0],
		target_xyz[:, 1],
		target_xyz[:, 2])

	target_estimate5 = cloud2cloud(source_xyz,
		source_val,
		target_xyz,
		stencil=5,
		verbose=True)

	target_estimate1 = cloud2cloud(source_xyz,
		source_val,
		target_xyz,
		stencil=1,
		limitsource=100000,
		verbose=True)

	max_error5 = np.max(np.abs(target_ref - target_estimate5))
	max_error1 = np.max(np.abs(target_ref - target_estimate1))

	assert max_error5 < 1e-5
	assert max_error1 < 1e-4


def test_n_dimensional():
	points_src = 100000
	points_tgt = 1000
	shp_val    = (3,4,5)
	x,y,z = np.random.rand(3, points_src)
	xi,yi,zi = np.random.rand(3, points_tgt)

	data = np.random.rand(points_src, *shp_val)

	base = CloudInterpolator((x,y,z), (xi,yi,zi))
	result = base.interp(data)

	assert result.shape == (points_tgt, *shp_val)


def test_function():
	SIZE_TEST= 20
	NVAR = 100
	K = 4

	xyz_source = np.random.rand(SIZE_TEST*1, SIZE_TEST*2, SIZE_TEST*3, 3)
	xyz_target = np.random.rand(SIZE_TEST*4, SIZE_TEST*5, 3)

	rad_source = np.linalg.norm(xyz_source, axis=-1)
	rad_tgt = np.linalg.norm(xyz_target, axis=-1)

	raw_data_array = np.repeat(np.expand_dims(rad_source, axis=-1), NVAR, axis=-1)

	function = lambda dists: np.power(dists, 4)

	out = cloud2cloud(xyz_source, raw_data_array, xyz_target, function=function, stencil=K, verbose=True)
	max_error = np.max(np.abs(out[:,:,0]-rad_tgt))

	assert max_error < 1e-1
