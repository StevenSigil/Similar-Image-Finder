from time import time


def st_time(func):
    """ Timer to test efficiency of functions.

    Use as @decorator before function or with syntax:
        `st_time(funcName, verbose)(funcVar1, funcVar2, ...)`

    `verbose` is expected to be 0 or 1 - 0 will print the 
    args/kwargs passed to child function
    """

    def st_func(*args, **kwargs):
        t1 = time()
        r = func(*args, **kwargs)
        t2 = time()
        print(f'\nFunction:\t"{func.__name__}()"\nTime:\t\t{t2 - t1}\n')
        return r

    return st_func
