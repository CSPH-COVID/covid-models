### Python Standard Library ###
import copy
import os
import datetime as dt
import json
import logging
import numpy as np
### Third Party Imports ###
from collections import OrderedDict
### Local Imports ###
from covid_model import CovidModel
from covid_model.runnable_functions import do_single_fit
from covid_model.utils import setup, get_filepath_prefix, db_engine


def remove_variants(model, y_dict, remove_variants):
    new_variant = [var for var in model.attrs['variant'] if var not in remove_variants][0]
    for variant in remove_variants:
        cmpts_to_remove = model.get_cmpts_matching_attrs({'variant': variant})
        for cmpt in cmpts_to_remove:
            new_cmpt = tuple(cmpt[:4] + tuple([new_variant]) + cmpt[5:])
            y_dict[new_cmpt] += y_dict[cmpt]
            del y_dict[cmpt]
    return y_dict


def main():
    ####################################################################################################################
    # Set Up Arguments for Running
    outdir = setup(os.path.basename(__file__), 'info')

    fit_args = {
        'batch_size': 5,
        'increment_size': 2,
        'tc_min': -0.99,
        'tc_max': 0.99,
        'forward_sim_each_batch': True,
        'look_back': None,
        'use_hosps_end_date': True,
        'window_size': 14,
        'last_window_min_size': 7,  # make this smaller because we are refitting anyways
        'write_batch_output': False,
        'outdir': outdir
    }
    model_args = {
        'params_defs': 'covid_model/analysis/20220525_model_simplification/params.json',
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/vacc_proj_params.json',
        'regions': ['co'],
        'tc': [0.75, 0.75],
        'tc_tslices': [14],
        'mobility_mode': None,
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-05-25', "%Y-%m-%d").date(),
        'max_step_size': np.inf
    }
    #model_args = {'base_spec_id': 2622}
    model_args = {}
    backtrack = 14
    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(json.dumps({"model_args": model_args}, default=str))
    logging.info(f'backtrack: {backtrack}')

    ####################################################################################################################
    # Run

    # fit a statewide model up to present day to act as a baseline
    logging.info('Fit all at once')
    #model = do_single_fit(**fit_args, **model_args)
    #model.solution_sum(['seir', 'variant', 'priorinf']).unstack().to_csv(get_file_prefix(outdir) + "states_seir_variant_priorinf_total_all_at_once.csv")

    logging.info('Fit in chunks')
    attrs = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                         'age': ['0-19', '20-39', '40-64', '65+'],
                         'vacc': ['none', 'shot1', 'shot2', 'shot3'],
                         'priorinf': ['none', 'omicron', 'other'],
                         'variant': ['none', 'alpha', 'delta'],
                         'immun': ['none', 'weak', 'strong'],
                         'region': ['co']})

    chunks = [
        {'end_date': dt.datetime.strptime('2021-10-01', '%Y-%m-%d').date(), 'variant': ['none', 'alpha', 'delta']},
        {'end_date': dt.datetime.strptime('2022-02-01', '%Y-%m-%d').date(), 'variant': ['alpha', 'delta', 'omicron', 'ba2']},
        {'end_date': dt.datetime.strptime('2022-05-15', '%Y-%m-%d').date(), 'variant': ['delta', 'omicron', 'ba2', 'ba2121']},
        {'end_date': dt.datetime.strptime('2022-06-01', '%Y-%m-%d').date(), 'variant': ['omicron', 'ba2', 'ba2121']}
    ]

    fit_args.update({'use_hosps_end_date': False})

    # do the fits
    new_model_args = copy.deepcopy(model_args)
    new_model_args.update({'end_date': chunks[0]['end_date'], 'attrs': copy.deepcopy(attrs)})
    new_model_args['attrs']['variant'] = chunks[0]['variant']
    #model = do_single_fit(**fit_args, **new_model_args, tags={'chunk': 1})
    #model = CovidModel(base_spec_id=2625)
    #model.prep()
    #model.solve_seir()
    #model.solution_sum(['seir', 'variant', 'priorinf']).unstack().to_csv(get_file_prefix(outdir) + f"states_seir_variant_priorinf_total_chunk{1}.csv")

    for j, chunk in enumerate(chunks[1:]):
        i = j + 2
        if i < len(chunks):
            continue

        model = CovidModel(base_spec_id=2630)
        model.prep()
        model.solve_seir()
        model.solution_sum_df(['seir', 'variant', 'priorinf']).unstack().to_csv(get_filepath_prefix(outdir) + f"states_seir_variant_priorinf_total_chunk{1}.csv")

        new_model_args = copy.deepcopy(model_args)
        new_model_args.update({'base_model': model, 'start_date': model.t_to_date(model.trange[-backtrack]), 'end_date': chunk['end_date'], 'attrs': copy.deepcopy(attrs)})
        new_model_args['attrs']['variant'] = chunk['variant']
        if i == len(chunks):
            fit_args.update({'use_hosps_end_date': True})  # last chunk runs to end of hosps
        y_final = {cmpt: y for cmpt, y in zip(model.compartments, model.solution_y[model.date_to_t(new_model_args['start_date']), :])}
        y_final = remove_variants(model, y_final, list(set(model.attrs['variant']) - set(chunk['variant'])))
        new_model_args.update({'y0_dict': y_final})

        model = do_single_fit(**fit_args, **new_model_args, tags={'chunk': i})
        model.solution_sum_df(['seir', 'variant', 'priorinf']).unstack().to_csv(get_filepath_prefix(outdir) + f"states_seir_variant_priorinf_total_chunk{i}.csv")


if __name__ == "__main__":
    main()