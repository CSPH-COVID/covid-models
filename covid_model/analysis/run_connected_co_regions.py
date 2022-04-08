### Python Standard Library ###
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
### Local Imports ###
from covid_model.db import db_engine
from covid_model import CovidModel
from covid_model.cli_specs import ModelSpecsArgumentParser


def run():
    # get fit params
    parser = ModelSpecsArgumentParser()
    specs_args = parser.specs_args_as_dict()


    # Model with no mobility
    dfs = []
    for mm in ['none', 'population_attached']:
        print(f"mobility_mode: {mm}")
        specs_args['mobility_mode'] = mm
        engine = db_engine()
        model = CovidModel(engine=engine, **specs_args)
        model.prep()
        model.solve_seir()
        dfs.append(model.solution_sum(['seir', 'region']).stack([0, 1]).reset_index(name='y').assign(mobility=mm))
    df = pd.concat(dfs)

    # plot
    #df['seir_age'] = df['seir'] + " : " + df['age']
    g = sns.FacetGrid(df, row='seir', hue='mobility', col='region', height=4, aspect=3, sharey=False)
    g.map(sns.lineplot, 't', 'y')
    g.add_legend()
    plt.tight_layout()
    plt.savefig("covid_model/output/connected_forwardsim_test.png")

    print("end")


if __name__ == '__main__':
    run()
