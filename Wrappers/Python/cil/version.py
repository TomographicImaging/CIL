version = '24.3.1.dev14+g75f7c4ec.d19800101'
major = 24
minor = 3
patch = 1
commit_hash = 'g75f7c4ec'
num_commit = 14
# work-around for https://github.com/pypa/setuptools_scm/issues/1059
if (commit_hash, num_commit) == ('None', 0):
    import re
    if (_v := re.search(r'\.dev(\d+)\+(\w+)', version)):
        num_commit, commit_hash = int(_v.group(1)), _v.group(2)
