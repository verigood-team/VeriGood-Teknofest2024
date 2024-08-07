import functools
import multiprocessing

def time_out(max_timeout):
    """Timeout decorator, parameter in seconds."""

    def timeout_decorator(item):
        """Wrap the original function."""

        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            try:
                pool = multiprocessing.Pool(processes=1)
                async_result = pool.apply_async(item, args=args, kwds=kwargs)
                return async_result.get(max_timeout)
            except Exception as e:
                return None

        return func_wrapper

    return timeout_decorator
