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
from args import parse_args

import os
import sys
import string
import shutil
import random
import logging
import threading

import numpy as np

args = None
logger = logging.getLogger(__name__)

ROOT = resources.files("pgm").parent
CONFIG = ROOT / "config"
TEST_DATA_DIR = ROOT / "datasets" / "tests"
TEST_DATA_DICT = {
    TEST_DATA_DIR / "standard" : lambda: None,
    TEST_DATA_DIR / "static" : lambda: None,
}

MIN_SEQ_SIZE = 50
MEANS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
VARS = [0.75, 0.5, 0.25, 0.05, 0.25, 0.5, 0.75]

# logging setup logging basicConfig is set.
# the cpp loggers are handed points to the logger info/debug/warning/error
# functions
def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s [%(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d | %H:%M:%S"
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

def get_next_state(n_states, prev_state):
    state = random.randint(0, n_states - 1)
    while prev_state == state:
        state = random.randint(0, n_states - 1)
    return state

# get standard seq test. generated sequences of values
# centered around floats specific to each state
def get_standard_seq_test(n_states, seed):
    rng = np.random.default_rng(seed=seed)
    alphabet = string.ascii_lowercase[:args.n_states]
    continue_prob = args.step_prob

    seq = []
    labs = []
    prev_state = -1

    while True:
        size = max(int(rng.normal(loc=args.step_n, scale=3)), 10)
        state = get_next_state(n_states, prev_state)
        prev_state = state

        seq += rng.normal(loc=MEANS[state], scale=VARS[state], size=size).tolist()
        labs.append(alphabet[state])

        if len(seq) > MIN_SEQ_SIZE:
            if continue_prob < random.random():
                break
            continue_prob *= args.step_prob
        
    return seq, labs

# get static seq test. generate sequences of values
# specific to each state (these are not random and just
# the same val over and over).
def get_static_seq_test(n_states, seed):
    rng = np.random.default_rng(seed=seed)
    alphabet = string.ascii_lowercase[:args.n_states]
    continue_prob = args.step_prob

    seq = []
    labs = []
    prev_state = -1

    while True: 
        size = max(int(rng.normal(loc=args.step_n, scale=3)), 10)
        state = get_next_state(n_states, prev_state)
        prev_state = state
        
        seq += [MEANS[state] for _ in range(size)]
        labs.append(alphabet[state])

        if len(seq) > MIN_SEQ_SIZE:
            if continue_prob < random.random():
                break
            continue_prob *= args.step_prob
    
    return seq, labs

# writes sequence generate from get_seq*_test
# to a file in dest_dir (usually TEST_DATA_DIR)
def write_to_file(seq, labs, fname, dest_dir):
    dataext = DataLoader.DATAFILE_EXTENSION
    dfile = dest_dir / f"{fname}{dataext}"
    
    with open(dfile, "w") as f:
        for elem in seq:
            f.write(f"{elem}\n")
    
    labext = DataLoader.LABFILE_EXTENSION
    labfile = dest_dir / f"{fname}{labext}"
    with open(labfile, "w") as f:
        for lab in labs:
            f.write(f"{lab}\n")

# generates directory tests by calling get_seq*_test
# functions and saving them to disk
def gen_dir_tests():
    seeds = random.sample(range(0, 10000), args.n_tests)
    logger.info(f"{len(seeds)=}")
    for i,test_data_dir in enumerate(TEST_DATA_DICT):
        write_dir = test_data_dir / f"{args.n_states}state"
        write_dir.mkdir(parents=True, exist_ok=True)
        
        seq_lens = []
        for seed in seeds:
            random.seed(seed)
            seq, labs = TEST_DATA_DICT[test_data_dir](args.n_states, seed)

            fname = f"{i}_{seed}"
            seq_lens.append(len(seq))

            write_to_file(seq, labs, fname, write_dir)

        mean_seq_len = sum(seq_lens) / len(seq_lens)
        min_seq_len = min(seq_lens)
        max_seq_len = max(seq_lens)

        logger.info(f"{write_dir=}")
        logger.info(f"{mean_seq_len=}")
        logger.info(f"{min_seq_len=}")
        logger.info(f"{max_seq_len=}")

# run the hmm load dir tests. 
def hmm_load_dir_tests():
    for i,test_data_dir in enumerate(TEST_DATA_DICT):
        hmm = HMM(
            n_states=args.n_states,
            ignore_warnings=args.ignore_warnings
        )
        model_name = f"{args.hmm_topology}_{args.n_states}state"
        hmm.load_from_config(CONFIG / model_name)
        
        write_dir = test_data_dir / f"{args.n_states}state"
        with open(write_dir / model_name, "w") as f:
            f.write("Initial HMM\n")
            f.write(str(hmm))
        
        data_loader = DataLoader(load_dir=write_dir)
        trainer = HMMTrainer(hmm, data_loader)
        
        trainer.initialize()
        trainer.train(50)
        
        with open(write_dir / model_name, "a") as f:
            f.write("Final HMM\n")
            f.write(str(hmm))

# main function
#   sets up logger and parses args
#   
#   if calling load-dir-test, loads test data dict (sets paths
#   and function pointers for data generation) and then runs
#   hmm_load_dir_tests.
def main():
    global args
    
    args = parse_args()
    random.seed(args.seed)
    setup_logger()
    
    if args.test_type == "hmm-load-dir-test":
        global MIN_SEQ_SIZE
        MIN_SEQ_SIZE = args.n_states * args.step_n

        load_test_data_dict()
        gen_dir_tests()
        hmm_load_dir_tests()

if __name__ == "__main__":
    main()

