from unittest import TestCase
from covid_model.db import db_engine, get_sqa_table
from covid_model.model_specs import CovidModelSpecifications
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *


class Test(TestCase):
    def test_get_sqa_table_specifications(self):
        engine = db_engine()
        table = get_sqa_table(engine, schema='covid_model', table='specifications')
        self.assertIsNotNone(table)

    def test_specifications_write_to_db(self):
        engine = db_engine()
        specs = CovidModelSpecifications.from_db(engine, 870)
        specs.write_to_db(engine)
