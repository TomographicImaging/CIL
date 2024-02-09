import ctypes
import platform
from ctypes import util
# check for the extension

if platform.system() == 'Linux':
    dll = 'libcilacc.so'
elif platform.system() == 'Windows':
    dll_file = 'cilacc.dll'
    dll = util.find_library(dll_file)
elif platform.system() == 'Darwin':
    dll = 'libcilacc.dylib'
else:
    raise ValueError('Not supported platform, ', platform.system())

cilacc = ctypes.cdll.LoadLibrary(dll)
