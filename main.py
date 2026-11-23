import argparse

from orangeoil.config import CFG
from orangeoil.db import get_connection


def build_parser():
    parser = argparse.ArgumentParser(prog="orangeoil")
    sub = parser.add_subparsers(dest="command", required=True)
    firms = sub.add_parser("firms")
    firms.add_argument("--start", default=CFG["start"])
    ais = sub.add_parser("ais")
    ais.add_argument("--seconds", type=int, default=0)
    sub.add_parser("map")
    sub.add_parser("train")
    sub.add_parser("forecast")
    return parser


def main():
    args = build_parser().parse_args()
    con = get_connection()
    if args.command == "firms":
        from orangeoil.firms import firms_sync

        firms_sync(con, args.start)
    elif args.command == "ais":
        from orangeoil.ais import run_collect

        run_collect(con, args.seconds)
    elif args.command == "map":
        from orangeoil.ais import map_tankers

        map_tankers(con)
    elif args.command == "train":
        from orangeoil.pipeline import run_train

        run_train(con)
    elif args.command == "forecast":
        from orangeoil.pipeline import run_forecast

        run_forecast(con)


if __name__ == "__main__":
    main()
