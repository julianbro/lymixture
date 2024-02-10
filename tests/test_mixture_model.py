"""
Test the functionality of the mixture model class.
"""
import unittest
from unittest import TestCase

from lymph.models import Unilateral

from lymixture import LymphMixtureModel

from fixtures import SIMPLE_SUBSITE, get_graph, get_patient_data


class TestMixtureModel(TestCase):
    """Unit test the mixture model class."""

    def setUp(self) -> None:
        """Create intermediate and helper objects for the tests."""
        self.model_cls = Unilateral
        self.num_components = 3
        self.patient_data = get_patient_data()
        self.graph_dict = get_graph(size="small")
        return super().setUp()


    def test_init(self):
        """Test the initialization of the mixture model."""
        model = LymphMixtureModel(
            model_cls=self.model_cls,
            model_kwargs={"graph_dict": self.graph_dict},
            num_components=self.num_components,
        )
        self.assertIsInstance(model, LymphMixtureModel)
        self.assertEqual(model.num_components, self.num_components)


    def test_load_patient_data(self):
        """Test the loading of patient data."""
        model = LymphMixtureModel(
            model_cls=self.model_cls,
            model_kwargs={"graph_dict": self.graph_dict},
            num_components=self.num_components,
        )
        model.load_patient_data(self.patient_data, split_by=SIMPLE_SUBSITE)
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
