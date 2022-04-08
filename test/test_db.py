### Python Standard Library ###
import datetime as dt
from unittest import TestCase
### Third Party Imports ###
### Local Imports ###
from covid_model.db import db_engine, get_sqa_table
from covid_model.model_specs import CovidModelSpecifications
from covid_model.model import CovidModel


class Test(TestCase):
    def test_get_sqa_table_specifications(self):
        engine = db_engine()
        table = get_sqa_table(engine, schema='covid_model', table='specifications')
        self.assertIsNotNone(table)

    def test_specifications_write_to_db(self):
        engine = db_engine()
        specs = CovidModelSpecifications(engine=engine, from_specs=1525)
        specs.write_to_db(engine)

    def test_model_write_results_to_db(self):
        engine = db_engine()
        model = CovidModel(engine=engine, from_specs=1525, end_date=dt.date(2020, 1, 31))
        model.prep()
        model.solve_seir()
        model.write_results_to_db(engine)
