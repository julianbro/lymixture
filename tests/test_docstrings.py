"""
Make doctests of the lymixture package discoverable by unittest.
"""
import doctest
import unittest

from lymixture import models
from lymixture import utils

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def load_tests(loader, tests: unittest.TestSuite, ignore):
    tests.addTests(doctest.DocTestSuite(models))
    tests.addTests(doctest.DocTestSuite(utils))
    return tests
