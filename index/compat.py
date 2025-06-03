import sys
from typing_extensions import TypedDict as ExtTypedDict

# Monkey patch typing.TypedDict to use typing_extensions.TypedDict
if sys.version_info < (3, 12):
    import typing
    typing.TypedDict = ExtTypedDict
    
    # Also patch the module to ensure imports work correctly
    sys.modules['typing'].TypedDict = ExtTypedDict
