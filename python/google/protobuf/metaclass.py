import sys

if sys.version_info[0] == 3:
    def decorator(cls):
        return cls.__metaclass__(
            cls.__name__, cls.__bases__, cls.__dict__.copy()
        )
else:
    def decorator(cls):
        return cls
