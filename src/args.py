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

    parser.add_argument(
        "-tt", "--test_type",
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
        "-ht", "--hmm-topology",
        type=str,
        choices=["bakis", "ergodic"],
        required=True,
        help=(
            "choose hmm topology. files are stored in ROOT/config/{args.hmm-topology}-{args.n-states} "
            "by default."
        )
    )
    parser.add_argument(
        "-nst", "--n-states",
        type=int,
        required=True,
        help="n states in the hmm and for test generation.",
    )
    parser.add_argument(
        "-sd", "--seed",
        type=int,
        default=434,
        help=(
            "seed for random generation. set at the beginning of the script for "
            "random. any further seeds are generated from this seed."
        )
    )
    parser.add_argument(
        "-nts", "--n-tests",
        type=int,
        default=100,
        help=(
            "n tests to run. generates seeds based on an intial seed detemined "
            "by args.seed."
        )
    )
    parser.add_argument(
        "-stp", "--step-prob",
        type=float_range(0.0, 1.0),
        default=0.95,
        help=(
            "probability of continuing to the next step. "
            "each generation step generates approximately args.step-n "
            "samples. squared after each step."
        )
    )
    parser.add_argument(
        "-stn", "--step-n",
        type=int,
        default=15,
        help=(
            "number of samples per step centered at random state i. "
            "the number of samples itself is normally distributed, centered "
            "at step-n."
        )
    )
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

    return parser.parse_args()

if __name__ == "__main__":
    pass

