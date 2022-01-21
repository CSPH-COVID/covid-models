import pandas as pd
import numpy as np


# tslices and values define a stepwise function; get the value of that function for a given t
def get_value_from_slices(tslices, values, t):
    if len(values) != len(tslices) - 1:
        raise ValueError(f"Length of values ({len(values)}) must equal length of tslices ({len(tslices)}) - 1.")
    for i in range(len(tslices)-1):
        if tslices[i] <= t < tslices[i+1]:
            # if i > len(values):
            #     raise ValueError(f"")
            return values[i]
    raise ValueError(f"Cannot fetch value from slices because t={t} is out of range.")


# recursive function to process parameters that include values for different time slices, and construct params for a specific t
def get_params(input_params, t, tslices=None):
    if type(input_params) == list:
        value = get_value_from_slices([0]+tslices+[99999], input_params, t)
        return get_params(value, t, tslices)
    elif type(input_params) == dict:
        if 'tslices' in input_params.keys():
            return get_params(input_params['value'], t, tslices=input_params['tslices'])
        else:
            return {k: get_params(v, t, tslices) for k, v in input_params.items()}
    else:
        return input_params


def calc_multipliers(multipliers, prevalences):
    flow = prevalences * multipliers
    combined_mult = flow.sum()
    next_prev = flow / combined_mult
    return combined_mult, next_prev


def calc_multiple_multipliers(transitions, multipliers, starting_prevalences):
    starting_prevalences = np.array(starting_prevalences)
    if starting_prevalences.sum() != 1:
        starting_prevalences = np.append(starting_prevalences, 1 - starting_prevalences.sum())
        multipliers = multipliers + [{label: 1 for label in multipliers[0].keys()}]
    mults = {}
    prevs = {transitions[0][0]: starting_prevalences}
    for fr, to, label in transitions:
        mults[label], prevs[to] = calc_multipliers(np.array([m[label] for m in multipliers]), np.array(prevs[fr]))

    return mults, {k: v[:-1].sum() for k, v in prevs.items()}
