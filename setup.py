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

import os
import subprocess
from setuptools import setup
from setuptools.command.build import build
from setuptools.command.editable_wheel import editable_wheel

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


if __name__ == "__main__":
    pass
