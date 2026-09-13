#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: setup.py
Author: Rohit Sridhar
Date: 02-09-2026
Last Modified: 
Version: X.X

Description:
    <Add Description>
"""

import sys
import os
import subprocess
import shutil
from setuptools import setup
from setuptools.command.build import build
from setuptools.command.editable_wheel import editable_wheel

BUILD_DIRS=["src/pgm", "src/pgm_project.egg-info"]
# BUILD_DIRS=["build/"]

def run_pgm_build():
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    script_root = os.path.abspath(os.path.join(setup_dir, "scripts"))

    script_path = os.path.join(script_root, "build.sh")

    print(f"--- Running Custom Build Script at {script_path} ---")
    subprocess.run([script_path], check=True)

# Hook for standard installs (pip install .)
class InstallPGMStandard(build):
    def run(self):
        run_pgm_build()
        super().run()

# Hook for editable installs (pip install -e .)
class InstallPGMEditable(editable_wheel):
    def run(self):
        run_pgm_build()
        super().run()

if any(arg in sys.argv for arg in ["wheel", "install", "develop", "editable", "editable_wheel", "build_ext", "bdist_wheel"]):
    print("[setup.py] Auto-purging stale build directories to avoid CMake conflicts...")
    for directory in BUILD_DIRS:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"  ✓ Cleaned: {directory}")
            except Exception as e:
                print(f"  ✗ Failed to clean {directory}: {e}")

# This tells setuptools to use your custom logic during installation
setup(
    cmdclass={
        "build": InstallPGMStandard,
        "editable_wheel": InstallPGMEditable,
    }
)

