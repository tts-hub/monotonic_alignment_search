# SPDX-FileCopyrightText: Enno Hermann
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tomli-w",
# ]
# ///

"""Add uv dependency constraints to the pyproject.toml file.

Usage:
$ .github/scripts/add_uv_constraints.py numpy==1.26.4 torch==2.2.2
"""

import sys
import tomllib
from pathlib import Path

import tomli_w

with Path("pyproject.toml").open("rb") as f:
    toml = tomllib.load(f)

if "constraint-dependencies" not in toml["tool"]["uv"]:
    toml["tool"]["uv"]["constraint-dependencies"] = []

for dep in sys.argv[1:]:
    toml["tool"]["uv"]["constraint-dependencies"].append(
        f"{dep}; platform_system == 'Linux'",
    )

with Path("pyproject.toml").open("wb") as f:
    tomli_w.dump(toml, f)
