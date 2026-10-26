import importlib.metadata
from packaging.version import Version
version = importlib.metadata.version("cil")

__v = Version(version)
major, minor, patch = __v.major, __v.minor, __v.micro
commit_hash = "" if __v.local is None else __v.local.split(".", 1)[0]
num_commit = __v.dev or 9_999
