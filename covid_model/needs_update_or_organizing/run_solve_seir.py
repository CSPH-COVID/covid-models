### Python Standard Library ###
import os
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
### Local Imports ###
from covid_model.utils import get_filepath_prefix
from covid_model import CovidModel, ModelSpecsArgumentParser, db_engine
from covid_model.data_imports import ExternalHosps


def run_solve_seir(outdir=None, model=None, tags={}, **specs_args):
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    engine = db_engine()
    if model is None:
        print("creating model")
        model = CovidModel(engine=engine, **specs_args)
    print("prepping model")
    model.prep()
    print("solving model")
    model.solve_seir()
    model.write_specs_to_db(engine=engine, tags={'regions': model.regions if model else specs_args['regions'],
                                                 'mobility_mode': model.mobility_mode if model else specs_args['mobility_mode'], **tags})
    model.write_results_to_db(engine=engine)

    df = model.solution_sum(['seir', 'region']).assign(**{'date':model.daterange}).set_index('date').stack([0, 1]).reset_index(name='y')
    dfh = model.solution_sum(['seir', 'region'], index_with_model_dates=True)['Ih'].stack('region').rename('modeled_hospitalized')
    hosps = pd.concat([ExternalHosps(engine=engine).fetch(county_ids=model.get_all_county_fips(region)).assign(region=region).set_index('region', append=True) for region in model.regions], axis=0)
    hosps.index.set_names('date', level=0, inplace=True)
    dfh = dfh.to_frame().join(hosps)
    dfh_melt = dfh.melt(value_vars=['currently_hospitalized', 'modeled_hospitalized'], ignore_index=False, var_name='model', value_name='hospitalized').set_index('model', append=True)

    if outdir:
        # plot
        print("plotting results")
        p = sns.relplot(data=df, x='date', y='y', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
        _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
        plt.savefig(f"{get_filepath_prefix(outdir)}{model.spec_id}_run_solve_seir_compartments.png", dpi=300)

        p = sns.relplot(data=dfh_melt, x='date', y='hospitalized', hue='model', col='region', col_wrap=min(3, len(model.regions)), kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
        _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
        plt.savefig(f"{get_filepath_prefix(outdir)}{model.spec_id}_run_solve_sier_hospitalized.png", dpi=300)

        print("saving results")
        df.to_csv(f"{get_filepath_prefix(outdir)}{model.spec_id}_run_solve_seir_compartments.csv")
        dfh.to_csv(f"{get_filepath_prefix(outdir)}{model.spec_id}_run_solve_seir_hospitalized.csv")
    return model, df, dfh


if __name__ == '__main__':
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))

    parser = ModelSpecsArgumentParser()
    specs_args = parser.specs_args_as_dict()

    _ = run_solve_seir(**specs_args, outdir=outdir)
