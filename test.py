# test.py

from src.engine.test_engine import run_test
from src.utils.args import parse_args


def main():
    args = parse_args()
    run_test(args.cfg)


if __name__ == "__main__":
    main()
