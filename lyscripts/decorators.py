"""
This module provides decorators that can be used to avoid repetitive snippets of code,
e.g. safely opening files or logging the state of a function call.
"""
import functools
import logging
from functools import wraps
from pathlib import Path
from typing import Any, BinaryIO, Callable, TextIO, Union


def extract_logger(*args, **kwargs) -> logging.Logger:
    """Extract a logger from the arguments if present.

    The function will first search if the function is in fact a method of a class that
    has any attributes that are instances of `logging.Logger`. If not, it will search
    the arguments and keyword arguments for instances of `logging.Logger` and return
    the first one it finds. If none is found, it will return a general logger.
    """
    first_arg = next(iter(args), None)
    attr_loggers = []
    if hasattr(first_arg, "__dict__"):
        for attr in first_arg.__dict__.values():
            if isinstance(attr, logging.Logger):
                attr_loggers.append(attr)

    return_args = []
    args_loggers = []
    for arg in args:
        if isinstance(arg, logging.Logger):
            args_loggers.append(arg)
        else:
            return_args.append(arg)

    return_kwargs = {}
    kwargs_loggers = []
    for key, value in kwargs.items():
        if isinstance(value, logging.Logger):
            kwargs_loggers.append(value)
        else:
            return_kwargs[key] = value

    found_loggers = [*attr_loggers, *args_loggers, *kwargs_loggers]
    logger = next(iter(found_loggers), None)

    if logger is None:
        logger = logging.getLogger("lyscripts")

    return logger, return_args, return_kwargs


def assemble_signature(*args, **kwargs) -> str:
    """Assemble the signature of the function call."""
    args_str = ", ".join(str(arg) for arg in args)
    kwargs_str = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    signature = ", ".join([args_str, kwargs_str])
    return signature


def log_state(
    direct_func: Callable = None,
    success_msg: str = None,
    logger: logging.Logger = None,
    log_level: int = logging.INFO,
) -> Callable:
    """Provide a decorator that logs the state of the function execution.

    This function can either be used directly as a decorator or be called with the
    desired `success_msg` to return a decorator that can be then in turn be used to
    decorate a function.
    """
    # pylint: disable=logging-fstring-interpolation
    def log_decorator(func: Callable):
        """The decorator wrapping the decorated function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """The wrapper around the decorated function."""
            found_logger, args, kwargs = extract_logger(*args, **kwargs)

            nonlocal logger
            if logger is None:
                logger = found_logger

            signature = assemble_signature(*args, **kwargs)
            logger.debug(f"Executing {func.__name__}({signature}).")

            try:
                result = func(*args, **kwargs)
                if success_msg is not None:
                    nonlocal log_level
                    logger.log(log_level, success_msg)
                return result

            except Exception as exc:
                logger.error(f"Error in {func.__name__}({signature}).", exc_info=exc)
                raise exc

        return wrapper

    if direct_func is not None:
        return log_decorator(direct_func)

    return log_decorator


def check_input_file_exists(loading_func: Callable) -> Callable:
    """Check if the file path provided to the `loading_func` exists."""
    @wraps(loading_func)
    def inner(file_path: str, *args, **kwargs) -> Any:
        """Wrapped loading function."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        return loading_func(file_path, *args, **kwargs)

    return inner


def provide_file(is_binary: bool) -> Callable:
    """Make sure a decorated function is provided with a file-like object.

    This means, the assembled decorator checks the argument type and, if necessary,
    opens the file to call the decorated function. The provided file is either a text
    file of - if `is_binary` is set to `True` - a binary file.
    """
    def assembled_decorator(loading_func: Callable) -> Callable:
        """Assembled decorator that provides the function with a text/binary file."""
        @wraps(loading_func)
        def inner(file_or_path: Union[str, Path, TextIO, BinaryIO], *args, **kwargs):
            """The wrapped function."""
            if isinstance(file_or_path, (str, Path)):
                file_path = Path(file_or_path)
                if not file_path.is_file():
                    raise FileNotFoundError(f"File {file_path} does not exist.")

                if is_binary:
                    with open(file_path, mode="rb") as bin_file:
                        return loading_func(bin_file, *args, **kwargs)
                else:
                    with open(file_path, mode="r", encoding="utf-8") as txt_file:
                        return loading_func(txt_file, *args, **kwargs)

            return loading_func(file_or_path, *args, **kwargs)

        return inner

    return assembled_decorator


def check_output_dir_exists(saving_func: Callable) -> Callable:
    """Make sure the parent directory of the saved file exists."""
    @wraps(saving_func)
    def inner(file_path: str, *args, **kwargs) -> Any:
        """Wrapped saving function."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return saving_func(file_path, *args, **kwargs)

    return inner
