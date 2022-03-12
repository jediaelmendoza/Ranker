import argparse
from pathlib import Path
import pandas as pd
import dataset_ranker as ds_ranker
import ranker as ranker
import numpy as np
import optuna as optuna


def ranking(args):
    ds_ranker.ranking(args)

def eval(args):
    ds_ranker.compute_apfdc(args)

def mean_apfdc(args):
    ds_ranker.collect_apfdc(args)

def add_dataset_parser_arguments(parser):
    parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    rank_parser = subparsers.add_parser(
        "rank",
        help="Rank dataset",
    )

    rank_parser.set_defaults(func=ranking)
    rank_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    rank_parser.add_argument(
        "-r",
        "--ranker",
        help="Specifies the ranker used on datasets pointwise or pairwise.",
        type=int,
        default=1,
    )

    rank_parser.add_argument(
        "-m",
        "--model",
        help="Specifies the model used on ranker decision tree/neural network.",
        type=int,
        default=1,
    )
    #Evaluate
    eval_parser = subparsers.add_parser(
        "eval",
        help="APFDc dataset",
    )

    eval_parser.set_defaults(func=eval)
    eval_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    #Mean all APFDcs
    mean_parser = subparsers.add_parser(
        "mean",
        help="Mean all APFDc dataset",
    )

    mean_parser.set_defaults(func=mean_apfdc)
    mean_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )

    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.unique_separator = "\t"
    args.func(args)

if __name__ == "__main__":
    main()

