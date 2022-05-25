import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt

def main():
    t0 = '2020-01-01'
    import json
    with open('params.json', 'r') as f:
        params = json.load(f)

    with open('attribute_multipliers.json') as f:
        mult = json.load(f)

    with open('timeseries_effects/multipliers.json') as f:
        tem = json.load(f)

    with open('timeseries_effects/mab_prevalence.csv') as f:
        df_mab = pd.read_csv(f)

    # approximate pax and mab with step functions
    df_mab['date'] = pd.to_datetime(df_mab['date'])

    mab_tslices = [
        dt.datetime.strptime('2020-01-24', '%Y-%m-%d'),
        dt.datetime.strptime('2021-10-02', '%Y-%m-%d'),
        dt.datetime.strptime('2021-10-09', '%Y-%m-%d'),
        dt.datetime.strptime('2021-10-16', '%Y-%m-%d'),
        dt.datetime.strptime('2021-10-23', '%Y-%m-%d'),
        dt.datetime.strptime('2021-11-01', '%Y-%m-%d'),
        dt.datetime.strptime('2021-11-10', '%Y-%m-%d'),
        dt.datetime.strptime('2021-12-17', '%Y-%m-%d'),
        dt.datetime.strptime('2021-12-26', '%Y-%m-%d'),
        dt.datetime.strptime('2023-01-01', '%Y-%m-%d'),
    ]

    t_index_rounded_down_to_tslices = pd.cut(df_mab['date'], mab_tslices, right=False, retbins=False, labels=mab_tslices[:-1])
    mab = df_mab.groupby(t_index_rounded_down_to_tslices).mean().reset_index('date')

    pax_tslices = [
        dt.datetime.strptime('2020-01-24', '%Y-%m-%d'),
        dt.datetime.strptime('2021-12-28', '%Y-%m-%d'),
        dt.datetime.strptime('2022-01-04', '%Y-%m-%d'),
        dt.datetime.strptime('2022-01-11', '%Y-%m-%d'),
        dt.datetime.strptime('2022-01-18', '%Y-%m-%d'),
        dt.datetime.strptime('2022-01-25', '%Y-%m-%d'),
        dt.datetime.strptime('2022-02-01', '%Y-%m-%d'),
        dt.datetime.strptime('2022-02-08', '%Y-%m-%d'),
        dt.datetime.strptime('2022-02-13', '%Y-%m-%d'),
        dt.datetime.strptime('2022-03-07', '%Y-%m-%d'),
        dt.datetime.strptime('2022-03-14', '%Y-%m-%d'),
        dt.datetime.strptime('2022-03-20', '%Y-%m-%d'),
        dt.datetime.strptime('2023-01-01', '%Y-%m-%d'),
    ]

    t_index_rounded_down_to_tslices = pd.cut(df_mab['date'], pax_tslices, right=False, retbins=False, labels=pax_tslices[:-1])
    pax = df_mab.groupby(t_index_rounded_down_to_tslices).mean().reset_index('date')

    f = plt.figure(figsize=(30,5))
    plt.step(df_mab['date'], df_mab['treatment_mab'], where='post')
    plt.step(mab['date'], mab['treatment_mab'], where='post')
    plt.savefig('mab_approx.png')

    f = plt.figure(figsize=(30,5))
    plt.step(df_mab['date'], df_mab['treatment_pax'], where='post')
    plt.step(pax['date'], pax['treatment_pax'], where='post')
    plt.savefig('pax_approx.png')

    pax = pax[['date', 'treatment_pax']].drop_duplicates('treatment_pax', keep='first')
    mab = mab[['date', 'treatment_mab']].drop_duplicates('treatment_mab', keep='first')

    pax['date'] = [d.date().strftime("%Y-%m-%d") for d in pax['date']]
    mab['date'] = [d.date().strftime("%Y-%m-%d") for d in mab['date']]


    pax_prev_v2 = [{'param': 'pax_prev', 'attrs': None, 'vals': pax.set_index('date').to_dict(), 'desc': ''}]
    mab_prev_v2 = [{'param': 'mab_prev', 'attrs': None, 'vals': mab.set_index('date').to_dict(), 'desc': ''}]
    pax_mult_v2 = [{'param': 'pax_hosp_adj', 'attrs': None, 'vals': {t0: tem['treatment_pax']['hosp']}, 'desc': ''}]
    mab_mult_v2 = [
        {'param': 'mab_hosp_adj', 'attrs': None, 'vals': {t0: tem['treatment_mab']['hosp']}, 'desc': ''},
        {'param': 'mab_hlos_adj', 'attrs': None, 'vals': {t0: tem['treatment_pax']['hosp']}, 'desc': ''}
    ]

    params_v2 = [{"param": param, "attrs": pd['attributes'], 'vals': {t: val for t, val in zip([t0] + pd['tslices'] if pd['tslices'] is not None else ['2020-01-01'], pd['values'] if isinstance(pd['values'], list) else [pd['values']])}, 'desc': ''} for param, param_list in params.items() for pd in param_list]
    mult_v2 = [{"param": am['param'], 'attrs': am['attrs'] if am.__contains__('attrs') else None, 'mults': {t0: am['mult'] if am.__contains__('mult') else None}, 'vals': {t0: am['val'] if am.__contains__('val') else None}, 'desc':am['desc']} for am in mult]
    for li in mult_v2:
        for key in ['vals', 'mults']:
            if li[key][t0] is None:
                del li[key]

    unified_params = params_v2 + mult_v2 + pax_prev_v2 + pax_mult_v2 + mab_prev_v2 + mab_mult_v2
    def format_li(li):
        s = f'{{"param": "{li["param"]}",\t"attrs": {json.dumps(li["attrs"])}'
        if li.__contains__('mults'):
            s += f',\t"mults": {json.dumps(li["mults"])}'
        if li.__contains__('vals'):
            s += f',\t"vals": {json.dumps(li["vals"])}'
        s += f',\t"desc": "{li["desc"]}"}}'
        return s
    pstring = "[\n  " + ",\n  ".join([format_li(li) for li in unified_params]) + "\n]"
    with open('unified_params.json', 'w') as f:
        f.write(pstring)

    print("done")






if __name__ == "__main__":
    main()