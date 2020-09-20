#!/usr/bin/env python
"""A module containing utility code such as timing and memory usage tracking.
Daniel S. Graham
"""

import functools
import time

def time_method(function_name=None):
    """Times the execution of a function as a decorator.

    Parameters
    __________
    function_name : str
        The name of the function to time.
        (default is None)

    Returns
    -------
        A decorator function.
    """
    def real_decorator(func):
        @functools.wraps(func)
        def wrapper_time_method(*args, **kwargs):
            t_start = time.time()
            result = func(*args, **kwargs)
            t_end = time.time()
            if function_name is None:
                name = func.__name__.upper()
            else:
                name = function_name
            elapsed_t = (t_end - t_start)
            print(f'TIMING: {name}'.ljust(40) + f'{elapsed_t:>39.4f}s')
            return result
        return wrapper_time_method
    return real_decorator
