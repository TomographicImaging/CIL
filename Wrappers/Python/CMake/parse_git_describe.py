import re, subprocess, sys, os

git_executable = os.path.abspath(sys.argv[1])

pattern = re.compile('v([0-9]*)\.([0-9]*)(\.*)([0-9]*)')
git_describe_string = subprocess.check_output(f'"{git_executable}" describe', shell=True).decode("utf-8").rstrip()
v = git_describe_string.split('-')
if len(v) == 3:
    git_version_string, git_commit_number, git_hash = git_describe_string.split('-')
elif len(v) == 1:
    git_version_string = v[0]
    git_commit_number = -1
    git_hash = -1

versions = pattern.match(git_version_string)

version = list(versions.groups())

major = minor = patch = ''
if version[0] != '':
    major = version[0]
if version[1] != '':
    minor = version[1]
if version[2] != '': 
    patch = version[3]
else:
    patch = 0
    
print (major, minor, patch, git_commit_number, git_hash)
