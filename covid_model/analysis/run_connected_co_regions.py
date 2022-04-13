### Python Standard Library ###
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
### Local Imports ###
from covid_model.db import db_engine
from covid_model import CovidModel
from covid_model.cli_specs import ModelSpecsArgumentParser
from covid_model.data_imports import ExternalHosps


def run():
    # get fit params
    parser = ModelSpecsArgumentParser()
    specs_args = parser.specs_args_as_dict()
    other_args = parser.non_specs_args_as_dict()

    # Model with different mobility modes
    dfs = []
    dfhs = []
    for mobility_mode in ['none', 'population_attached']:
        specs_args['mobility_mode'] = mobility_mode
        engine = db_engine()
        print("creating model")
        model = CovidModel(engine=engine, **specs_args)
        model.write_specs_to_db(engine=engine, tags={'regions': specs_args['regions'], 'mobility_mode': specs_args['mobility_mode']})
        print("prepping model")
        model.prep()
        print("solving model")
        model.solve_seir()
        model.write_results_to_db(engine=engine)
        dfs.append(model.solution_sum(['seir', 'region']).assign(**{'date':model.daterange}).set_index('date').stack([0, 1]).reset_index(name='y').assign(mobility=specs_args['mobility_mode']))
        dfhs.append(model.solution_sum(['seir', 'region'], index_with_model_dates=True)['Ih'].stack('region').rename(mobility_mode))
    hosps = pd.concat([ExternalHosps(engine=engine).fetch(county_ids=model.get_all_county_fips(region)).assign(region=region).set_index('region', append=True) for region in model.regions], axis=0)
    hosps.index.set_names('date', level=0, inplace=True)

    df = pd.concat(dfs).reset_index(drop=True)
    dfh = pd.concat(dfhs, axis=1).join(hosps)

    # plot
    print("plotting results")
    #df['seir_age'] = df['seir'] + " : " + df['age']
    p = sns.relplot(data=df, x='date', y='y', hue='mobility', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig("covid_model/output/connected_forwardsim_compartments.png", dpi=300)

    dfh_melt = dfh.melt(value_vars=['currently_hospitalized', 'none', 'population_attached'], ignore_index=False, var_name='mode', value_name='hospitalized').set_index('mode', append=True)
    p = sns.relplot(data=dfh_melt, x='date', y='hospitalized', hue='mode', col='region', col_wrap=3, kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig("covid_model/output/connected_forwardsim_hospitalized.png", dpi=300)

    df.to_csv("covid_model/output/connected_forwardsim_compartments.csv")
    dfh_melt.to_csv("covid_model/output/connected_forwardsim_hospitalized.csv")





if __name__ == '__main__':
    run()
