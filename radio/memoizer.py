
_CACHE = dict()

class memoize:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        global _CACHE
        # Create a key based on args and kwargs
        key = (args, frozenset(kwargs.items()))

        if key in _CACHE:
            return _CACHE[key]

        result = self.func(*args, **kwargs)
        _CACHE[key] = result
        return result

    @staticmethod
    def clear_cache():
        global _CACHE
        _CACHE.clear()

    def __get__(self, obj, objtype):
        """Support instance methods."""
        from functools import partial
        return partial(self.__call__, obj)
