import numpy as np

def random_radius (inner_radius, outer_radius, *, rng=np.random):
    return np.exp(rng.uniform(low=np.log(inner_radius), high=np.log(outer_radius)))

def random_point_on_sphere (shape, *, rng=np.random):
    while True:
        p = rng.randn(*shape)
        p_norm_squared = np.sum(np.square(p))
        if p_norm_squared >= 1.0e-10:
            return p / np.sqrt(p_norm_squared)

def random_point_with_uniform_radial_distribution (*, shape, inner_radius, outer_radius, rng=np.random):
    return random_radius(inner_radius, outer_radius, rng=rng)*random_point_on_sphere(shape=shape, rng=rng)

if __name__ == '__main__':
    inner_radius = 1.0e-8
    outer_radius = 1.0e-2
    # c = normalizing_constant(inner_radius, outer_radius)

    for _ in range(100000):
        p = random_point_on_sphere((3,))
        norm_p = np.sqrt(np.sum(np.square(p)))
        assert (norm_p - 1.0) < 1.0e-10

    shape = (3,)
    point_v = [random_point_with_uniform_radial_distribution(shape=shape, inner_radius=inner_radius, outer_radius=outer_radius) for _ in range(100000)]
    r_v = [np.sqrt(np.sum(np.square(point))) for point in point_v]
    log_r_v = np.log(r_v)
    bin_v,bin_edge_v = np.histogram(log_r_v, bins=10, range=(np.log(inner_radius), np.log(outer_radius)))
    print('bin_v = {0}'.format(bin_v))

    print('test passed')
