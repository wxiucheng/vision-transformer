# train.py

from src.utils.args import parse_args
from src.engine.train_engine import run_train


def main():
    args = parse_args()
    run_train(args.cfg)


if __name__ == "__main__":
    main()
