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

from pgm.models import HMM
from pgm.utils.data_loader import DataLoader
from pgm.trainer import HMMTrainer
from pgm.utils import cpp_logger

import os
import sys
import logging
import threading
import argparse

import numpy as np

args = None
logger = logging.getLogger(__name__)

ROOT = resources.files("pgm").parent
TEST_DATA_DIR = ROOT / "datasets" / "tests"

# parse args for script
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
        "-act", "--action",
        type=str,
        choices=["gen-dir-tests", "hmm-seq-test", "hmm-load-dir-test"],
        help=(
            "action for script. gen-dir-tests generates directory tests (data files to load)."
            "hmm-seq-test runs a 1d hmm on just a few generated data points."
            "hmm-load-dir-test runs a 1d hmm on the generated tests from gen-dir-tests."
            "gen-dir-tests are typically stored TEST_DATA_DIR."
        )
    )
    parser.add_argument(
        "-dbg", "--debug",
        action="store_true",
        help="run in debug mode. more verbose logging and other things.",
    )

    return parser.parse_args()

# logging setup logging basicConfig is set.
# the cpp loggers are handed points to the logger info/debug/warning/error
# functions
def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d | %H:%M:%S'
    )

    cpp_logger.set_info_logger(logger.info)
    cpp_logger.set_debug_logger(logger.debug)
    cpp_logger.set_warning_logger(logger.warning)
    cpp_logger.set_error_logger(logger.error)

# these generate lists for each of the seq tests.
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

# runs hmm seq tests (using generated tests
# above
def run_hmm_seq_tests():
    l_final = get_static_seq_test()
    obs = np.array(l_final)

    hmm = HMM(2, False)
    hmm.initialize_model()

    logger.info("#################### Training HMMs: ####################")
    logger.info("########################################")
    for i in range(20):
        hmm.baum_welch(obs)
        if i % 5 == 0:
            logger.info(f"#################### Iter {i} ####################")
        logger.info("########################################")
    logger.info("#################### After 20 its: ####################")
    logger.info("########################################")

# writes sequence generate from get_seq*_test
# toa  file in dest_dir (usually TEST_DATA_DIR)
def write_seq_to_file(seq, fname, dest_dir):
    with open(dest_dir / fname, "w") as f:
        for elem in seq:
            f.write(f"{elem}\n")

# generates directory tests by calling get_seq*_test
# functions and saving them to disk
def gen_dir_tests():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    seq_0 = get_seq_test()
    seq_1 = get_static_seq_test()
    seq_2 = get_alternating_test()

    write_seq_to_file(seq_0, "0.txt", TEST_DATA_DIR)
    write_seq_to_file(seq_1, "1.txt", TEST_DATA_DIR)
    write_seq_to_file(seq_2, "2.txt", TEST_DATA_DIR)

def main():
    global args
    args = parse_args()
    setup_logger()
    
    if args.action == "gen-dir-tests":
        gen_dir_tests()
    elif args.action == "run-hmm-dir-tests":
        run_hmm_dir_tests()
    elif args.action == "run-hmm-seq-tests":
        run_hmm_seq_tests()

if __name__ == "__main__":
    main()

