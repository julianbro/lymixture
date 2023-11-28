"""
.. include:: ../README.md
"""
import argparse
import logging
import re

from rich.containers import Lines
from rich.logging import RichHandler
from rich.text import Text
from rich_argparse import RichHelpFormatter

from lyscripts import app, data, evaluate, plot, predict, sample, temp_schedule
from lyscripts._version import version
from lyscripts.utils import report

__version__ = version
__description__ = "Package containing scripts used in lynference pipelines"
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lyscripts"

# nopycln: file


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RichDefaultHelpFormatter(
    RichHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """
    Empty class that combines the functionality of displaying the default value with
    the beauty of the `rich` formatter
    """
    def _rich_fill_text(self, text: Text, width: int, indent: Text) -> Text:
        text_cls = type(text)
        if text[0] == text_cls("\n"):
            text = text[1:]

        paragraphs = text.split(separator="\n\n")
        text_lines = Lines()
        for par in paragraphs:
            no_newline_par = text_cls(" ").join(line for line in par.split())
            wrapped_par = no_newline_par.wrap(self.console, width)

            for line in wrapped_par:
                text_lines.append(line)

            text_lines.append(text_cls("\n"))

        return text_cls("\n").join(indent + line for line in text_lines) + "\n\n"


RichDefaultHelpFormatter.styles["argparse.syntax"] = "red"
RichDefaultHelpFormatter.styles["argparse.formula"] = "green"
RichDefaultHelpFormatter.highlights.append(
    r"\$(?P<formula>[^$]*)\$"
)
RichDefaultHelpFormatter.styles["argparse.bold"] = "bold"
RichDefaultHelpFormatter.highlights.append(
    r"\*(?P<bold>[^*]*)\*"
)
RichDefaultHelpFormatter.styles["argparse.italic"] = "italic"
RichDefaultHelpFormatter.highlights.append(
    r"_(?P<italic>[^_]*)_"
)


def exit_cli(args: argparse.Namespace):
    """Exit the cmd line tool"""
    if args.version:
        report.print("lyscripts ", __version__)
    else:
        report.print("No command chosen. Exiting...")


def main():
    """
    Utility for performing common tasks w.r.t. the inference and prediction tasks one
    can use the `lymph` package for.
    """
    parser = argparse.ArgumentParser(
        prog="lyscripts",
        description=re.sub(r"\s+", " ", main.__doc__)[1:],
        formatter_class=RichDefaultHelpFormatter,
    )
    parser.set_defaults(run_main=exit_cli)
    parser.add_argument(
        "-v", "--version", action="store_true",
        help="Display the version of lyscripts"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    subparsers = parser.add_subparsers()

    # the individual scripts add `ArgumentParser` instances and their arguments to
    # this `subparsers` object
    app._add_parser(subparsers, help_formatter=parser.formatter_class)
    data._add_parser(subparsers, help_formatter=parser.formatter_class)
    evaluate._add_parser(subparsers, help_formatter=parser.formatter_class)
    plot._add_parser(subparsers, help_formatter=parser.formatter_class)
    predict._add_parser(subparsers, help_formatter=parser.formatter_class)
    sample._add_parser(subparsers, help_formatter=parser.formatter_class)
    temp_schedule._add_parser(subparsers, help_formatter=parser.formatter_class)

    args = parser.parse_args()

    handler = RichHandler(
        console=report,
        show_time=False,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(args.log_level)

    args.run_main(args)
