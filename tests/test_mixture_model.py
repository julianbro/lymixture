"""
Test the functionality of the mixture model class.
"""
import unittest
import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
from fixtures import MixtureModelFixture
from lymph.models import Unilateral

from lymixture import LymphMixture
from lymixture.utils import RESP_COL

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class TestMixtureModel(MixtureModelFixture, TestCase):
    """Unit test the mixture model class."""

    def setUp(self) -> None:
        """Create intermediate and helper objects for the tests."""
        self.setup_rng(seed=42)
        self.setup_mixture_model(
            model_cls=Unilateral,
            num_components=3,
            graph_size="small",
            load_data=True,
        )
        self.setup_responsibilities()
        return super().setUp()


    def test_init(self):
        """Test the initialization of the mixture model."""
        self.assertIsInstance(self.mixture_model, LymphMixture)
        self.assertEqual(len(self.mixture_model.components), self.num_components)


    def test_load_patient_data(self):
        """Test the loading of patient data."""
        total_num_patients = 0
        for subgroup in self.mixture_model.subgroups.values():
            total_num_patients += len(subgroup.patient_data)

        self.assertEqual(total_num_patients, len(self.patient_data))


    def test_assign_responsibilities(self):
        """Test the assignment of responsibilities."""
        self.mixture_model.set_responsibilities(self.resp)

        stored_resp = np.empty(shape=(0, self.num_components))
        for subgroup in self.mixture_model.subgroups.values():
            self.assertIn(RESP_COL, subgroup.patient_data)
            stored_resp = np.vstack([
                stored_resp, subgroup.patient_data[RESP_COL].to_numpy()
            ])
        np.testing.assert_array_equal(self.resp, stored_resp)
        stored_resp = self.mixture_model.patient_data[RESP_COL]
        np.testing.assert_array_equal(self.resp, stored_resp)
        stored_resp = self.mixture_model.get_responsibilities()
        np.testing.assert_array_equal(self.resp, stored_resp)


    def test_get_responsibilities(self):
        """Test accessing the responsibilities."""
        self.mixture_model.set_responsibilities(self.resp)
        p_idx = self.rng.integers(low=0, high=len(self.patient_data))
        c_idx = self.rng.integers(low=0, high=self.num_components)
        self.assertEqual(
            self.resp[p_idx,c_idx],
            self.mixture_model.get_responsibilities(patient=p_idx, component=c_idx)
        )


if __name__ == "__main__":
    unittest.main()
