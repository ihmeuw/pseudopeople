""" This script updates the README.rst file with the latest information about
the project. It is intended to be run from the github "update README" workflow.
"""

import json
import re

from packaging.version import parse

# Load supported python versions
with open("python_versions.json", "r") as f:
    versions = json.load(f)
versions_str = ", ".join(versions)
versions = [parse(v) for v in versions]
max_version = max(versions).base_version

# Open README.md and replace the line containing the python versions

# Replace python versions in readme
with open("README.rst", "r") as file:
    readme = file.read()
readme = re.sub(r"python=\d+\.\d+", "python=" + max_version, readme)
readme = re.sub(
    r"Supported Python versions: .*", r"Supported Python versions: " + versions_str, readme
)

# Write the updated README.md back to file
with open("README.rst", "w") as file:
    file.write(readme)
