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

from pathlib import Path
from importlib import resources

from pgm import models.HMM
from pgm import utils.data_loader.DataLoader
from pgm import utils.cpp_logger
# from pgm.models import HMM
# from pgm.utils.data_loader import DataLoader
# from pgm.utils import cpp_logger

import os
import sys
import logging
import threading
import argparse

import numpy as np

args = None
logger = logging.getLogger(__name__)

ROOT = Path(resources.files("pgm")._paths[0]).parent.parent
TEST_DATA_DIR = ROOT / "datasets" / "tests"

def parse_args():
    """
    parse_args
    parses args passed via command line
    """
    parser = argparse.ArgumentParser(
        description="run-pgm options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-gts", "--gen-tests",
        action="store_true",
        help="generate tests using numpy random number generator. for now, generates them in datasets/tests.",
    )
    parser.add_argument(
        "-dbg", "--debug",
        action="store_true",
        help="run in debug mode. more verbose logging and other things.",
    )

    return parser.parse_args()

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d | %H:%M:%S'
    )
    cpp_logger.set_info_logger(logger.info)

def get_seq_test():
    rng = np.random.default_rng(seed=42)
    l1 = rng.normal(loc=0.0, scale=0.5, size=10).tolist()
    l2 = rng.normal(loc=2.0, scale=1.0, size=30).tolist()
    return l1 + l2

def get_static_seq_test():
    l1 = [1.0 for _ in range(10)]
    l2 = [0.0 for _ in range(30)]
    return l1 + l2

def get_alternating_test():
    l_final = []
    for i in range(100):
        l_final.append(0.0 if i % 2 == 0 else 1.0)
    return l_final

def run_hmm_seq_tests():
    l_final = get_static_seq_test()
    obs = np.array(l_final)

    hmm = HMM(2, False)
    hmm.initialize_model()

    logger.info("#################### Initialization: ####################")
    logger.info(hmm)
    logger.info("########################################")
    for i in range(20):
        hmm.baum_welch(obs)
        if i % 5 == 0:
            logger.info(f"#################### Iter {i} ####################")
            logger.info(hmm)
        logger.info("########################################")
    logger.info("#################### After 20 its: ####################")
    logger.info(hmm)
    logger.info("########################################")

def gen_tests():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_loader = DataLoader(load_dir=TEST_DATA_DIR)
    data_loader.load_data()

def main():
    global args
    args = parse_args()
    setup_logger()
    
    if args.gen_tests:
        gen_tests()
    else:
        data_loader = DataLoader(load_dir=Path("/data/hmm_modeling/fs_pgm/datasets"))
        data_loader.load_data()

if __name__ == "__main__":
    main()

