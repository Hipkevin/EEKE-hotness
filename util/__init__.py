from functools import wraps
import time

def timer(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        begin_time = time.perf_counter()
        result = func(*args, **kwargs)
        start_time = time.perf_counter()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (func.__name__, args, kwargs, start_time - begin_time))
        return result

    return wrap