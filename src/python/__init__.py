#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: __init__.py
Author: Rohit Sridhar
Date: 14-09-2026
Last Modified: 
Version: X.X

Description:
    <Add Description>
"""

from . import _pgm
import sys

# Register the submodules globally into sys.modules under their absolute import paths
sys.modules["pgm.models"] = _pgm.models
sys.modules["pgm.utils"] = _pgm.utils
sys.modules["pgm.utils.cpp_logger"] = _pgm.utils.cpp_logger
sys.modules["pgm.utils.data_loader"] = _pgm.utils.data_loader

# expose models and utils
models = _pgm.models
utils = _pgm.utils
