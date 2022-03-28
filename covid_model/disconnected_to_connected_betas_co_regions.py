import json
import datetime as dt
from functools import reduce
from collections import OrderedDict
from os.path import exists
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from db import db_engine
from covid_model import RegionalCovidModel, all_regions
from covid_model.cli_specs import ModelSpecsArgumentParser
from analysis.charts import actual_hosps, modeled


def run():
    # get fit params
    parser = ModelSpecsArgumentParser()
    parser.add_argument("-rg", "--regions", nargs="+", choices=all_regions.keys(), required=True, help="Specify the regions to be run, default is all regions (not counties)")
    parser.add_argument("-sids", "--spec_ids", nargs="+", type=int, required=True, help="Specify the spec_ids corresponding to each region (must match the order of regions)")
    parser.add_argument("-rp", "--region_params", type=str, default="input/region_params.json", help="the path to the region-specific params file to use for fitting; default to 'input/region_params.json'")
    fit_params = parser.parse_args()
    regions = fit_params.regions
    spec_ids = fit_params.spec_ids
    spec_id_dict = OrderedDict(zip(regions, spec_ids))
    with open(fit_params.region_params, 'r') as f:
        region_params = json.loads(f.read())
    engine = db_engine()

    # retrieve beta and In for each run
    beta_dfs = []
    In_dfs = []
    for region, spec_id in [(region, spec_id_dict[region]) for region in regions]:
        fnames = ["output/connected_regional_model/cache/" + str(spec_id) + "_" + region + "_In.csv",
                  "output/connected_regional_model/cache/" + str(spec_id) + "_" + region + "_beta.csv",
                  "output/connected_regional_model/cache/" + str(spec_id) + "_" + region + "_solution.csv"]
        if not all([exists(fname) for fname in fnames]):
            print(f'Retrieving Model Results for {region} using spec_id: {spec_id_dict[region]}')
            region_args = parser.specs_args_as_dict()
            region_args.update({'from_specs': spec_id_dict[region]})
            model = RegionalCovidModel(engine=engine, region=region, **region_args)
            model.prep()
            model.solve_seir()
            solution = model.solution_sum('seir')[['S', 'E', 'A', 'I', 'Ih', 'D']]
            In = (model.model_params['lamb'] * solution['A'] + solution['I']) / region_params[region]['total_pop']
            In_df = pd.DataFrame(index=model.daterange, columns=[region], data=In.values)
            beta_df = pd.DataFrame(index=model.tslices_dates, columns=[region], data=[(1 - tc) * model.model_params['betta'] for tc in model.tc])
            In_df.to_csv(fnames[0])
            beta_df.to_csv(fnames[1])
            solution.to_csv(fnames[2])
            actual_hosps(engine, county_ids=region_params[region]['county_fips'])
            modeled(model, 'Ih')
            plt.savefig('output/connected_regional_model/' + str(spec_id) + '_fitted_hosps_'  + region + ".png", dpi=300)
            plt.close()
        else:
            print(f'Retrieving Model Results for {region} from output/connected_regional_model/cache directory')
            In_df = pd.read_csv(fnames[0], index_col=0, parse_dates=True)
            beta_df = pd.read_csv(fnames[1], index_col=0, parse_dates=True)
        beta_dfs.append(beta_df)
        In_dfs.append(In_df)
    beta_df = pd.concat(beta_dfs, axis=1)
    # fill in the gaps of beta_df
    beta_df = pd.DataFrame(index=pd.date_range(min(beta_df.index), max(beta_df.index))).join(beta_df).fillna(method='ffill').fillna(0)
    #In_df   = pd.concat(In_dfs, axis=1).dropna()
    In_df   = pd.concat(In_dfs, axis=1).fillna(0)
    start_date = max(min(beta_df.index), min(In_df.index))
    end_date = min(max(beta_df.index), max(In_df.index))

    beta_df = beta_df.loc[(beta_df.index >= start_date) & (beta_df.index <= end_date)]
    In_df = In_df.loc[(In_df.index >= start_date) & (In_df.index <= end_date)]

    # build contact matrix
    print(f'Building Contact Matrix')
    fname = "output/connected_regional_model/cache/cm.pkl"
    if not exists(fname):
        cm = RegionalCovidModel.construct_region_contact_matrices(OrderedDict([(region, all_regions[region]) for region in regions]), region_params, fpath='../contact_matrices/mobility.csv')
        pkl.dump(cm, open(fname, 'wb'))
    else:
        cm = pkl.load(open(fname, 'rb'))

    ### Solve for connected betas, need to retrieve the correct mobility matrix for each model tslice date
    # population attached beta
    mobility_dates = pd.Series(cm['dwell_matrices'].keys())

    print(f'Creating Plots')
    betaps = []
    betars = []
    for i, date in enumerate(In_df.index):
        cm_i = cm['dwell_matrices'][mobility_dates[np.where(mobility_dates <= date)[0][-1]]]
        Dtilde = cm_i['dwell_rownorm']
        Dstar = cm_i['dwell_colnorm']
        beta_i = np.squeeze(np.array(beta_df.loc[beta_df.index == date]))
        In_i = np.squeeze(np.array(In_df.loc[In_df.index == date]))
        betaps.append(beta_i * In_i / np.linalg.multi_dot([Dtilde, np.transpose(Dstar), In_i]))
        betars.append(np.dot(np.linalg.inv(Dtilde), beta_i * In_i) / np.dot(np.transpose(Dstar), In_i))

    betap_df = pd.DataFrame(index=In_df.index, columns=In_df.columns, data=np.stack(betaps))
    betar_df = pd.DataFrame(index=In_df.index, columns=In_df.columns, data=np.stack(betars))

    In_df.to_csv("output/connected_regional_model/In_df")
    beta_df.to_csv("output/connected_regional_model/beta_df")
    betap_df.to_csv("output/connected_regional_model/betap_df")
    betar_df.to_csv("output/connected_regional_model/betar_df")

    nr = int(np.ceil(np.sqrt(len(regions))))
    nc = int(np.ceil(len(regions)/nr))
    f, axs = plt.subplots(nr, nc, figsize=(5*nr, 5*nc), sharex=True, sharey=False)
    for i, region in enumerate(regions):
        #axs.flat[i].step(beta_df.index, beta_df[region], where='post', color=(0.1, 0.1, 0.1))
        beta_df[region].plot(ax=axs.flat[i], color=(0.1, 0.1, 0.1))
        betap_df[region].plot(ax=axs.flat[i], color='b')
        betar_df[region].plot(ax=axs.flat[i], color='r')
        axs.flat[i].set_title(all_regions[region])
        axs.flat[i].legend(['Disconnected', 'Option 1', 'Option 2'])
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("output/connected_regional_model/betas_comparison.png")
    plt.close()

    # for each region, plot some details of its betas, and the contact with other regions
    In_df.columns = [all_regions[reg] for reg in In_df.columns]
    for i, region in enumerate(regions):
        f, axs = plt.subplots(4, 1, figsize=(25, 20))
        f.suptitle(all_regions[region])
        axs.flat[0].set_title("Beta")
        axs.flat[1].set_title("Infectius Contact With Other Regions")
        axs.flat[2].set_title("Contact With Other Regions")
        axs.flat[3].set_title("Percent Infectious of Other Regions")
        # betas
        beta_df[region].plot(ax=axs.flat[0], color=(0.1, 0.1, 0.1))
        betap_df[region].plot(ax=axs.flat[0], color='b')
        betar_df[region].plot(ax=axs.flat[0], color='r')
        axs.flat[0].legend(['Disconnected', 'Option 1', 'Option 2'])

        # contact with infectious from other regions
        region_contact = []
        region_In_contact = []
        for j, date in enumerate(In_df.index):
            cm_j = cm['dwell_matrices'][mobility_dates[np.where(mobility_dates <= date)[0][-1]]]
            In_j = In_df.iloc[j]
            region_contact.append(np.dot(cm_j['dwell_rownorm'], np.transpose(cm_j['dwell_colnorm']))[i, :])
            region_In_contact.append(region_contact[-1] * In_j)
        region_contact = pd.DataFrame(np.stack(region_contact), index=In_df.index, columns=In_df.columns)
        region_In_contact = pd.DataFrame(np.stack(region_In_contact), index=In_df.index, columns=In_df.columns)

        region_In_contact.drop(all_regions[region], axis=1).plot(ax=axs.flat[1])
        region_contact.drop(all_regions[region], axis=1).plot(ax=axs.flat[2])
        In_df.drop(all_regions[region], axis=1).plot(ax=axs.flat[3])
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("output/connected_regional_model/mobility_detailed_" + region)
        plt.close()

        # plot disconnected vs Option 1 only
        f = plt.figure(figsize=(20, 15), dpi=300)
        ax = plt.gca()
        plt.title("Mobility Adjustments, Population Attached Beta: " + all_regions[region])
        beta_df[region].plot(ax=ax, color=(0.1, 0.1, 0.1))
        betap_df[region].plot(ax=ax, color='b', style='--')
        f.tight_layout()
        plt.savefig("output/connected_regional_model/mobility_option1_" + region)

        # plot disconnected vs Option 1 only
        f = plt.figure(figsize=(20, 15), dpi=300)
        ax = plt.gca()
        plt.title("Mobility Adjustments, Location Attached Beta: " + all_regions[region])
        beta_df[region].plot(ax=ax, color=(0.1, 0.1, 0.1))
        betar_df[region].plot(ax=ax, color='r', style='--')
        f.tight_layout()
        plt.savefig("output/connected_regional_model/mobility_option2_" + region)



if __name__ == '__main__':
    run()
