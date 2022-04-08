import json

from matplotlib import pyplot as plt
import seaborn as sns

from covid_model.db import db_engine
from covid_model import CovidModel, RegionalCovidModel
from covid_model.model_fit import CovidModelFit
from covid_model.cli_specs import ModelSpecsArgumentParser


def run():
    # get fit params
    parser = ModelSpecsArgumentParser()
    specs_args = parser.specs_args_as_dict()

    # build / solve model
    engine = db_engine()
    model = CovidModel(engine=engine, **specs_args)
    model.prep()
    model.solve_seir()

    # plot
    df = model.solution_sum(['seir', 'region']).stack([0, 1]).reset_index(name='y')
    #df['seir_age'] = df['seir'] + " : " + df['age']
    g = sns.FacetGrid(df, row='seir', column='region', height=4, aspect=3, sharey=False)
    g.map(sns.lineplot, 't', 'y')
    g.add_legend()
    plt.tight_layout()
    plt.savefig("covid_model/output/connected_forwardsim_test.png")


    print("end")


if __name__ == '__main__':
    run()
