import numpy as np


def filter_vector(a):
    raise NotImplementedError("recalculate indices")


def my_cosine_dist(a, b):
    return np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-8)


metrics = {
    "cos_sim": {
        "measure": my_cosine_dist, #NOTNOTUSE scipy.spatial.distance.cosine-THIS-is-(1-cos(a,b))
        "is_symetric": True
    },
    "dot": {
        "measure": np.dot,
        "is_symetric": True
    },
    "euclidian": {
        "measure": lambda a, b: np.linalg.norm(a-b),
        "is_symetric": True
    }
}


def get_metric_func(name):
    return metrics[name]["measure"]
