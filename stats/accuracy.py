import numpy as np

def avg_euclidean_dst(points_from, points_to):
    """Computes average euclidean distance between the provided points.
    Provided pointues may represent points in multidimensional space.

    Parameters:
        points_from, points_to --- numpy arrays of source and target points

    Returns:
        average Euclidean distance between points.

    Raises:
        ValueError if input arrays have different shape
    """
    if (points_from.shape != points_to.shape):
        raise ValueError("Point arrays should have equal shape")
    num_points = points_from.shape[-1]
    points_from = points_from.transpose()
    points_to = points_to.transpose()
    sum_dst = 0
    for point_from, point_to in zip(points_from, points_to):
        sum_dst += np.sqrt(np.sum((point_from - point_to)**2))
    return sum_dst / num_points
