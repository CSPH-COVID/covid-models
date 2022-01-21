from covid_model.data_imports import ExternalVacc, ExternalHosps
from covid_model.model_specs import CovidModelSpecifications
from covid_model.model_fit import CovidModelFit
from covid_model.model import CovidModel


class MetroCovidModelSpecifications(CovidModelSpecifications):
    def __init__(self):
        super().__init__()

    def set_actual_vacc(self, engine):
        self.actual_vacc_df = ExternalVacc(engine, t0_date=self.start_date).fetch(('08001', ))


class MetroCovidModelFit(CovidModelFit):
    def set_actual_hosp_from_db(self, engine):
        self.actual_hosp = ExternalHosps(engine, t0_date=self.base_specs.start_date).fetch('emresource_hosps.csv')['currently_hospitalized']


class MetroModel(CovidModel):
    @property
    def y0_dict(self):
        y0d = {('S', age, 'unvacc'): n for age, n in self.specifications.group_pops.items()}
        y0d[('I', '40-64', 'unvacc')] = 2.2
        y0d[('S', '40-64', 'unvacc')] -= 2.2
        return y0d

