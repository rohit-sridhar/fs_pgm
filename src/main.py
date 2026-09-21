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
import random
import logging
import threading
import argparse

import numpy as np

args = None
logger = logging.getLogger(__name__)

ROOT = resources.files("pgm").parent
TEST_DATA_DIR = ROOT / "datasets" / "tests"
TEST_DATA_DICT = {
    TEST_DATA_DIR / "standard" : lambda: None,
    TEST_DATA_DIR / "static" : lambda: None,
    TEST_DATA_DIR / "alternating": lambda: None,
}

MEANS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
VARS = [0.75, 0.5, 0.25, 0.05, 0.25, 0.5, 0.75]
NUM_VALUES = [40, 30, 20, 10, 20, 30, 40]
SEEDS = [32, 43, 56, 21, 78]

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
        choices=["hmm-load-dir-test"],
        required=True,
        help=(
            "action for script. gen-dir-tests generates directory tests (data files to load)."
            "hmm-seq-test runs a 1d hmm on just a few generated data points."
            "hmm-load-dir-test runs a 1d hmm on the generated tests from gen-dir-tests."
            "gen-dir-tests are typically stored TEST_DATA_DIR."
        )
    )
    parser.add_argument(
        "-nst", "--n-states",
        type=int,
        required=True,
        help="n states in the hmm and for test generation.",
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

# load test data dict
def load_test_data_dict():
    global TEST_DATA_DICT
    TEST_DATA_DICT[TEST_DATA_DIR / "standard"] = get_standard_seq_test
    TEST_DATA_DICT[TEST_DATA_DIR / "static"] = get_static_seq_test
    TEST_DATA_DICT[TEST_DATA_DIR / "alternating"] = get_alternating_seq_test

# these generate lists for each of the seq tests.
def get_standard_seq_test(n_states, seed):
    rng = np.random.default_rng(seed=seed)
    seq = []
    for i in range(n_states):
        seq += rng.normal(loc=MEANS[i], scale=VARS[i], size=NUM_VALUES[i]).tolist()
    return seq

def get_static_seq_test(n_states, seed):
    seq = []
    rng = np.random.default_rng(seed=seed)
    for i in range(n_states):
        size = int(rng.normal(loc=NUM_VALUES[i], scale=1.0))
        seq += [MEANS[i] for _ in range(size)]
    return seq

def get_alternating_seq_test(n_states, seed):
    seq = []
    size = sum(NUM_VALUES[:n_states])
    for i in range(size):
        ri = random.randint(0, n_states - 1)
        seq.append(MEANS[ri])
    return seq

# writes sequence generate from get_seq*_test
# toa  file in dest_dir (usually TEST_DATA_DIR)
def write_seq_to_file(seq, fname, dest_dir):
    with open(dest_dir / fname, "w") as f:
        for elem in seq:
            f.write(f"{elem}\n")

# generates directory tests by calling get_seq*_test
# functions and saving them to disk
def gen_dir_tests():
    for i,test_data_dir in enumerate(TEST_DATA_DICT):
        test_data_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            seq = TEST_DATA_DICT[test_data_dir](args.n_states, seed)
            ext = DataLoader.DATAFILE_EXTENSION
            write_seq_to_file(seq, f"{i}_{seed}{ext}", test_data_dir)

def hmm_load_dir_tests():
    gen_dir_tests()

    for i,test_data_dir in enumerate(TEST_DATA_DICT):
        if i >= 1:
            break
        hmm = HMM(n_states=args.n_states)
        hmm.initialize_model()
        with open(test_data_dir / "hmm.txt", "w") as f:
            f.write(str(hmm))

        data_loader = DataLoader(load_dir=test_data_dir)
        trainer = HMMTrainer(hmm, data_loader)

        trainer.train(30)
        with open(test_data_dir / "hmm.txt", "a") as f:
            f.write(str(hmm))

def main():
    global args
    args = parse_args()

    setup_logger()
    load_test_data_dict()
    
    if args.action == "hmm-load-dir-test":
        hmm_load_dir_tests()

if __name__ == "__main__":
    main()

