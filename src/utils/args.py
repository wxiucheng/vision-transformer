# src/utils/args.py

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--cfg",
            type = str,
            default = None,
            help = "YAML配置文件路径",
            )

    return parser.parse_args()
