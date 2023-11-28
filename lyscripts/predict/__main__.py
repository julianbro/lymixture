import argparse

from lyscripts import RichDefaultHelpFormatter, exit_cli
from lyscripts.predict import prevalences, risks

# I need another __main__ guard here, because otherwise pdoc tries to run this
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="lyscripts predict",
        description=__doc__,
        formatter_class=RichDefaultHelpFormatter,
    )
    parser.set_defaults(run_main=exit_cli)
    subparsers = parser.add_subparsers()

    # the individual scripts add `ArgumentParser` instances and their arguments to
    # this `subparsers` object
    risks._add_parser(subparsers, help_formatter=parser.formatter_class)
    prevalences._add_parser(subparsers, help_formatter=parser.formatter_class)

    args = parser.parse_args()
    args.run_main(args)
