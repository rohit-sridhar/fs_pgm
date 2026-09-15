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

# pre set up housekeeping. clean and recreate build dir
build_dir = os.path.join(os.path.dirname(__file__), "pgm")

if any(arg in sys.argv for arg in ["wheel", "install", "develop", "editable", "editable_wheel", "build_ext", "bdist_wheel"]):
    print("[setup.py] Auto-purging stale build directories to avoid CMake conflicts...")
    if os.path.exists(build_dir):
        try:
            shutil.rmtree(build_dir)
            print(f"  ✓ Cleaned: {build_dir}")
        except Exception as e:
            print(f"  ✗ Failed to clean {build_dir}: {e}")
    os.makedirs(build_dir, exist_ok=True)


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

# This tells setuptools to use your custom logic during installation
setup(
    cmdclass={
        "build": InstallPGMStandard,
        "editable_wheel": InstallPGMEditable,
    }
)

