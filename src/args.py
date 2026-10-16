#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: src/args.py
Author: Rohit Sridhar
Date: 23-09-2026
Last Modified:
Version: X.X

Description:
    argument handler for main.py
        exports a parse_args function that is used with
        src/main.py (run-pgm from shell when installed)
"""

import argparse
import sys

HMM_TOPOLOGIES=["bakis", "ergodic", "composite"]

REQUIRED_BY = {
    "": {},
    "hmm-load-dir-test": {"hmm-topology", "n-states"},
    "gen-plots": {"hmm-topology", "n-states"},
    "test-mv-gaussian": {"n-states"},
}

def is_required(arg):
    test_type = ""
    
    for i,opt in enumerate(sys.argv):
        if opt in ["-tt", "--test-type"]:
            test_type = sys.argv[i + 1]
    
    return arg in REQUIRED_BY[test_type]

# takes args, defines a custom float range checking function based on 
# the passed args and returns the function pointer
def float_range(min_val, max_val, incl_lower=False, incl_upper=False):
    """Return a function that validates a float within a specific range."""
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < min_val or f > max_val:
            raise argparse.ArgumentTypeError(f"must be in range [{min_val} .. {max_val}]")
        return f

    return float_range_checker

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
    main_args = parser.add_argument_group("Main Args", "Main arguments for script")
    modeling_args = parser.add_argument_group("Modeling Args", "Modeling arguments for hyperparameter and configuration")
    test_args = parser.add_argument_group("Test Args", "Arguments for configuration of test generation (static and standard norm)")
    boolean_args = parser.add_argument_group("Booleans Args", "Boolean arguments for script config (debug, spread, etc.)")

    # main script options
    main_args.add_argument(
        "-sd", "--seed",
        type=int,
        default=434,
        help=(
            "seed for random generation. set at the beginning of the script for "
            "random. any further seeds are generated from this seed."
        )
    )

    main_args.add_argument(
        "-tt", "--test-type",
        type=str,
        choices=["hmm-load-dir-test", "gen-plots", "test-mv-gaussian"],
        required=True,
        help=(
            "action for script. gen-dir-tests generates directory tests (data files to load)."
            "hmm-seq-test runs a 1d hmm on just a few generated data points."
            "hmm-load-dir-test runs a 1d hmm on the generated tests from gen-dir-tests."
            "gen-dir-tests are typically stored TEST_DATA_DIR."
        )
    )

    # modeling hyperparameters
    modeling_args.add_argument(
        "-ht", "--hmm-topology",
        type=str,
        choices=HMM_TOPOLOGIES,
        required=is_required("hmm-topology"),
        help=(
            "choose hmm topology. files are stored in ROOT/config/{args.hmm-topology}-{args.n-states} "
            "by default."
        )
    )
    modeling_args.add_argument(
        "-nst", "--n-states",
        type=int,
        required=is_required("n-states"),
        help="n states in the hmm and for test generation.",
    )

    # Test generation args
    test_args.add_argument(
        "-nts", "--n-tests",
        type=int,
        default=5,
        help=(
            "n tests to generate. Sets a seed for each test and saves"
            "test data in datasets/tests/[test_type]/[n-state]/[seed]"
            "The generator seed for each test is seeded from args.seed"
        )
    )
    test_args.add_argument(
        "-tsz", "--test-size",
        type=int,
        default=10,
        help=(
            "generate test_size samples per test. generates seeds "
            "based on an intial seed detemined by args.seed."
        )
    )
    test_args.add_argument(
        "-stn", "--step-n",
        type=int,
        default=2,
        help=(
            "number of samples per step centered at random state i. "
            "the number of samples itself is normally distributed, centered "
            "at step-n."
        )
    )
    test_args.add_argument(
        "-stp", "--step-prob",
        type=float_range(0.0, 1.0),
        default=0.95,
        help=(
            "probability of continuing to the next step. "
            "each generation step generates approximately args.step-n "
            "samples. squared after each step."
        )
    )

    # Boolean flags
    parser.add_argument(
        "-dbg", "--debug",
        action="store_true",
        help="run in debug mode. more verbose logging and other things.",
    )
    parser.add_argument(
        "-igw", "--ignore-warnings",
        action="store_true",
        help=(
            "ignore warnings. won't log warnings wrapped around this bool. "
            "enables debug/info level logging without warnings cluttering output."
        ),
    )
    parser.add_argument(
        "-spd", "--spread",
        action="store_true",
        help="use spread means (multiplies means by scalar but keeps vars the same).",
    )

    return parser.parse_args()

if __name__ == "__main__":
    pass

