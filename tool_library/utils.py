def is_ray_remote_function(func):
    return hasattr(func, "remote") and callable(getattr(func, "remote"))
