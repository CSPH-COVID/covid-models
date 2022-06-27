""" Python Standard Library """
from unittest import TestCase
""" Third Party Imports """
import numpy as np
""" Local Imports """
from covid_model.model_fit import CovidModelFit
from covid_model.model import CovidModel
from covid_model.db import db_engine


class Test(TestCase):
    def setUp(self) -> None:
        self.engine = db_engine()
        self.fit = CovidModelFit(engine=self.engine, from_specs=1036)
        self.fit.set_hosp(self.engine)

    def test_run(self):
        self.fit.run(self.engine, look_back=1)
        model1 = self.fit.fitted_model

        model2 = CovidModel(base_model=model1)
        model2.solve_seir()

        for t in model1.trange:
            self.assertEqual(model1.params_by_t[t], model2.params_by_t[t])
            # np.testing.assert_array_equal(model1.linear_matrix[t], model2.linear_matrix[t])
            # np.testing.assert_array_equal(model1.nonlinear_matrices[t], model2.nonlinear_matrices[t])
            # np.testing.assert_array_equal(model1.constant_vector[t], model2.constant_vector[t])
            # np.testing.assert_array_equal(model1.nonlinear_multiplier[t], model2.nonlinear_multiplier[t])

        np.testing.assert_almost_equal(model1.solution_y, model2.solution_y)
