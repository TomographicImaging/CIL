import importlib.metadata
from packaging.version import Version
version = importlib.metadata.version("cil")

__v = Version(version)
major = __v.major
minor = __v.minor
patch = __v.micro

commit_hash = __v.local
num_commit = __v.dev
