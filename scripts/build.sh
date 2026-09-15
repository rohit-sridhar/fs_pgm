#!/usr/bin/env bash
# -*- coding: utf-8 -*-
set -euo pipefail

# File: build.sh
# Date: 01-09-2026
# Last Modified:

# Description
#

# 1. Create a separate directory for build artifacts to keep your project clean
# mkdir -p pgm
cd pgm

# 2. Configure the project, passing the nanobind path
cmake .. -Dnanobind_DIR="/usr/local/lib/python3.12/site-packages/nanobind/cmake"

# 3. Compile the code
cmake --build . --config Release

