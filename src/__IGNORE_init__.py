#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: src/__init__.py
Author: Rohit Sridhar
Date: 12-09-2026
Last Modified: 
Version: X.X

Description:
    <Add Description>
"""

import sys
from . import _pgm

# 1. Expose top-level components to the pgm package namespace
utils = _pgm.utils
models = _pgm.models

# 2. Inject them into sys.modules so 'from pgm import models' works
sys.modules["pgm.utils"] = utils
sys.modules["pgm.models"] = models

# 3. Inject nested submodules so 'from pgm.utils import cpp_logger' works
sys.modules["pgm.utils.cpp_logger"] = utils.cpp_logger
sys.modules["pgm.utils.data_loader"] = utils.data_loader

# 4. Control what gets imported if someone runs 'from pgm import *'
__all__ = ["utils", "models"]

