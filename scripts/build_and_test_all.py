import os
import subprocess
import re
import tempfile
import platform

os.environ["PYTHONUNBUFFERED"] = "1"

# Define Python and NumPy version combinations
combinations = {
    "3.10": ["1.23", "1.24", "1.25", "1.26"],
    "3.11": ["1.23", "1.24", "1.25", "1.26"],
    "3.12": ["1.26"],
}

# Define channels and paths
channels = "-c conda-forge -c https://software.repos.intel.com/python/conda -c ccpi"
anaconda_user = "ccpi"
is_linux = platform.system() == "Linux"
script_dir = os.path.dirname(os.path.abspath(__file__))
recipe_path = os.path.join(script_dir, '..', 'recipe')
test_path = os.path.join(script_dir, '..', 'Wrappers', 'Python', 'test')
env_file = os.path.join(script_dir, 'requirements-test.yml')

def extract_packages_and_paths(packages_out):
    """Extract package names and their paths from the conda build output."""
    lines = packages_out.strip().split("\n")
    return {
        os.path.basename(line).replace(".tar.bz2", ""): os.path.dirname(line)
        for line in lines
    }

def remove_environment(env_name):
    """Remove a conda environment if it exists."""
    check_env_command = f"conda env list | grep -w {env_name}"
    remove_env_command = f"conda env remove -n {env_name} -y --quiet"

    print(f"Checking if environment {env_name} exists...")
    check_result = subprocess.run(check_env_command, shell=True, text=True, capture_output=True, check=False)
    if check_result.returncode == 0:
        print(f"Environment {env_name} exists. Removing it...")
        subprocess.run(remove_env_command, shell=True, text=True, check=True)
    else:
        print(f"Environment {env_name} does not exist. Skipping removal.")


def create_environment(env_name, package_name):
    """Create a conda environment from a temporary environment file."""
    temp_env_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yml")
    try:
        # Copy and filter the environment file
        with open(env_file, "r", encoding="utf-8") as original, open(temp_env_file.name, "w", encoding="utf-8") as temp:
            for line in original:
                if not is_linux and "# [linux]" in line:
                    continue
                temp.write(line)

        # Append the package to the environment file
        with open(temp_env_file.name, "a", encoding="utf-8") as temp:
            temp.write(f"  - local::{package_name}\n")

        # Create the environment
        create_env_command = f"conda env create -f {temp_env_file.name} -n {env_name}"
        print(f"Running: {create_env_command}")
        subprocess.run(create_env_command, shell=True, text=True, check=True)
    finally:
        os.unlink(temp_env_file.name)

def run_tests(env_name, python_version, numpy_version):
    """Run tests in the specified environment."""
    test_command = f"conda run -n {env_name} --no-capture-output python -m unittest discover -q -b -s {test_path}"
    print(f"Running: {test_command}", flush=True)

    result = subprocess.run(test_command, shell=True, text=True, capture_output=True, check=True)
    stderr_output = result.stderr

    match = re.search(r"Ran (\d+) tests in ([\d.]+)s\n\n(OK|FAILED)(.*)", stderr_output, re.DOTALL)
    if match:
        tests_run = int(match.group(1))
        time_taken = match.group(2)
        status = match.group(3)
        skipped_match = re.search(r"skipped=(\d+)", match.group(4))
        skipped = int(skipped_match.group(1)) if skipped_match else 0

        return {
            "python_version": python_version,
            "numpy_version": numpy_version,
            "tests_run": tests_run,
            "skipped": skipped,
            "time_taken": time_taken,
            "status": status,
        }
    return None

def print_test_summary(test_results):
    """Print a summary of the test results."""
    print("\n=== Test Summary ===")
    print(f"{'Python Version':<15} {'NumPy Version':<15} {'Tests Run':<10} {'Skipped':<10} {'Time Taken':<15} {'Status':<10}")
    print("=" * 80)
    for test in test_results:
        print(f"{test['python_version']:<15} {test['numpy_version']:<15} {test['tests_run']:<10} {test['skipped']:<10} {test['time_taken']:<15} {test['status']:<10}")

def build_and_test():
    """Build once for each Python version, then test for each NumPy version."""
    command = f"conda build {recipe_path} {channels}"

    # Generate the package names and paths
    print(f"Running: {command} --output")
    result = subprocess.run(f"{command} --output", shell=True, text=True, capture_output=True, check=True)
    packages = extract_packages_and_paths(result.stdout)

    # Build and test each package
    test_results = []
    for package, path in packages.items():
        match = re.search(r"py(\d+)", package)
        if not match:
            raise ValueError(f"Could not extract Python version from package name: {package}")
        
        python_version_digits = match.group(1)
        python_version = f"{python_version_digits[0]}.{python_version_digits[1:]}"
        numpy_versions = combinations[python_version]

        if not os.path.exists(os.path.join(path, f"{package}.tar.bz2")):
            # build the package if it doesn't exist
            rebuild_command = f"{command} --no-test --no-anaconda-upload --python {python_version}"
            print(f"Running: {rebuild_command}")
            subprocess.run(rebuild_command, shell=True, text=True, check=True)
        else:
            print(f"Package {package} already built at {path}.")

        for numpy_version in numpy_versions:
            print(f"\n=== Testing Python {python_version} and NumPy {numpy_version} ===")

            package_name = package.replace("-", "=")
            np_version_digits = numpy_version.replace(".", "")
            env_name = f"test_py{python_version_digits}_np{np_version_digits}"

            remove_environment(env_name)
            create_environment(env_name, package_name)

            result = run_tests(env_name, python_version, numpy_version)
            if result:
                test_results.append(result)

            remove_environment(env_name)

    print_test_summary(test_results)

    package_upload = input("Do you want to upload the package? (y/n): ")
    if package_upload.lower() == "y":
        for package, path in packages.items():
            upload_command = f"anaconda upload {path} --user {anaconda_user}"
            print(f"Running: {upload_command}")
            subprocess.run(upload_command, shell=True, text=True, check=True)
    else:
        print("Skipping upload.")

if __name__ == "__main__":
    build_and_test()