import warnings
from functools import wraps


def deprecate(new_function_name: str = ""):
    def inner_decorator_function(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            warnings.warn(
                "get_require_spellCheck is deprecated"
                + "; use has_spelling_correction"
                if new_function_name
                else "",
                DeprecationWarning,
            )
            return func(*args, **kwargs)

        return wrapper_func

    return inner_decorator_function
