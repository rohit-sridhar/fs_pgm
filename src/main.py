#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: main.py
Author: Rohit Sridhar
Date: 01-09-2026
Last Modified: 
Version: X.X

Description:
    <Add Description>
"""

from pgm.hmm import HMM

import numpy as np

def main():
    obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    hmm = HMM(4)
    hmm.initialize_model()

    hmm.baum_welch(obs)

if __name__ == "__main__":
    main()

