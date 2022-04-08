### Python Standard Library ###
import json
import datetime as dt
from collections import OrderedDict
from os.path import exists
import pickle as pkl
### Third Party Imports ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
### Local Imports ###
from covid_model.db import db_engine
from covid_model import RegionalCovidModel
from covid_model.cli_specs import ModelSpecsArgumentParser
from covid_model.analysis.charts import actual_hosps, modeled


def run():
    # get fit params
    parser = ModelSpecsArgumentParser()
    parser.add_argument("-rg", "--regions", nargs="+", choices=all_regions.keys(), required=True, help="Specify the regions to be run, default is all regions (not counties)")
    parser.add_argument("-sids", "--spec_ids", nargs="+", type=int, required=True, help="Specify the spec_ids corresponding to each region (must match the order of regions)")
    parser.add_argument("-rp", "--region_definitions", type=str, default="input/region_definitions.json", help="the path to the region-specific params file to use for fitting; default to 'input/region_definitions.json'")
    parser.add_argument("-mob", "--mobility", type=str, help="the file path to a mobility file; default to fetching mobility data from the database")
    fit_params = parser.parse_args()
    regions = fit_params.regions
    spec_ids = fit_params.spec_ids
    spec_id_dict = OrderedDict(zip(regions, spec_ids))
    with open(fit_params.region_definitions, 'r') as f:
        region_definitions = json.loads(f.read())
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
            In = (model.model_params['lamb'] * solution['A'] + solution['I']) / region_definitions[region]['total_pop']
            In_df = pd.DataFrame(index=model.daterange, columns=[region], data=In.values)
            beta_df = pd.DataFrame(index=model.tslices_dates, columns=[region], data=[(1 - tc) * model.model_params['betta'] for tc in model.tc])
            In_df.to_csv(fnames[0])
            beta_df.to_csv(fnames[1])
            solution.to_csv(fnames[2])
            actual_hosps(engine, county_ids=region_definitions[region]['county_fips'])
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
        cm = RegionalCovidModel.construct_region_contact_matrices(OrderedDict([(region, all_regions[region]) for region in regions]), region_definitions, engine=engine, fpath=fit_params.mobility)
        pkl.dump(cm, open(fname, 'wb'))
    else:
        cm = pkl.load(open(fname, 'rb'))

    ### Solve for connected betas, need to retrieve the correct mobility matrix for each model tslice date
    print(f'Solving for Connected Betas')
    mobility_dates = pd.Series(cm['dwell_matrices'].keys())

    betaps = []
    betals = []
    for i, date in enumerate(In_df.index):
        cm_i = cm['dwell_matrices'][mobility_dates[np.where(mobility_dates <= date)[0][-1]]]
        Dtilde = cm_i['dwell_rownorm']
        Dstar = cm_i['dwell_colnorm']
        beta_i = np.squeeze(np.array(beta_df.loc[beta_df.index == date]))
        In_i = np.squeeze(np.array(In_df.loc[In_df.index == date]))
        betaps.append(beta_i * In_i / np.linalg.multi_dot([Dtilde, np.transpose(Dstar), In_i]))
        betals.append(np.dot(np.linalg.inv(Dtilde), beta_i * In_i) / np.dot(np.transpose(Dstar), In_i))

    betap_df = pd.DataFrame(index=In_df.index, columns=In_df.columns, data=np.stack(betaps))
    betal_df = pd.DataFrame(index=In_df.index, columns=In_df.columns, data=np.stack(betals))

    In_df.to_csv("output/connected_regional_model/In_df.csv")
    beta_df.to_csv("output/connected_regional_model/beta_df.csv")
    betap_df.to_csv("output/connected_regional_model/betap_df.csv")
    betal_df.to_csv("output/connected_regional_model/betal_df.csv")

    In_df.columns = [all_regions[reg] for reg in In_df.columns]

    print(f'Creating Plots')
    tstart = dt.datetime.strptime("2020-04-12", "%Y-%m-%d")
    # create color map
    colors = sns.color_palette("husl", 11)
    colordict = dict(zip(In_df.columns, colors))

    # plot the betas of all the regions together
    nr = int(np.ceil(np.sqrt(len(regions))))
    nc = int(np.ceil(len(regions)/nr))
    f, axs = plt.subplots(nr, nc, figsize=(5*nr, 5*nc), sharex=True, sharey=False)
    for i, region in enumerate(regions):
        beta_df[region].loc[beta_df.index >= tstart].plot(ax=axs.flat[i], color=(0.1, 0.1, 0.1))
        betap_df[region].loc[betap_df.index >= tstart].plot(ax=axs.flat[i], color='b', style='--')
        betal_df[region].loc[betal_df.index >= tstart].plot(ax=axs.flat[i], color='r', style='--')
        axs.flat[i].set_title(all_regions[region])
        axs.flat[i].legend(['Disconnected', 'Pop Attached', 'Loc Attached'])
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("output/connected_regional_model/betas_comparison_both_options.png")
    plt.close()

    f, axs = plt.subplots(nr, nc, figsize=(5 * nr, 5 * nc), sharex=True, sharey=False)
    for i, region in enumerate(regions):
        beta_df[region].loc[beta_df.index >= tstart].plot(ax=axs.flat[i], color=(0.1, 0.1, 0.1))
        betap_df[region].loc[betap_df.index >= tstart].plot(ax=axs.flat[i], color='b', style='--')
        axs.flat[i].set_title(all_regions[region])
        axs.flat[i].legend(['Disconnected', 'Pop Attached', 'Loc Attached'])
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("output/connected_regional_model/betas_comparison_option1.png")
    plt.close()

    f, axs = plt.subplots(nr, nc, figsize=(5 * nr, 5 * nc), sharex=True, sharey=False)
    for i, region in enumerate(regions):
        beta_df[region].loc[beta_df.index >= tstart].plot(ax=axs.flat[i], color=(0.1, 0.1, 0.1))
        betal_df[region].loc[betal_df.index >= tstart].plot(ax=axs.flat[i], color='r', style='--')
        axs.flat[i].set_title(all_regions[region])
        axs.flat[i].legend(['Disconnected', 'Loc Attached'])
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("output/connected_regional_model/betas_comparison_option2.png")
    plt.close()

    # plot prevalences for each year separately
    for year in [2020, 2021, 2022]:
        f = plt.figure(figsize=(15, 8), dpi=300)
        ax = plt.gca()
        plt.title("Prevalence of All Regions")
        In_df.iloc[In_df.index.year == year].plot(ax=ax, color=[colordict[x] for x in In_df.columns])
        f.tight_layout()
        plt.savefig("output/connected_regional_model/prevalence_" + str(year))

    # for each region, plot some details of its betas, and the contact with other regions
    for i, region in enumerate(regions):
        name_prefix = str(spec_id_dict[region]) + "_" + region
        f, axs = plt.subplots(5, 1, figsize=(15, 40))
        f.suptitle(all_regions[region])
        axs.flat[0].set_title("Relative Prevalence To This Region")
        axs.flat[1].set_title("Beta")
        axs.flat[2].set_title("This Region's Contact With Other Regions (Anywhere)")
        axs.flat[3].set_title("Fraction of Time Spent In This Region Done By People From Other Regions")
        axs.flat[4].set_title("Fraction of Other Region's Time Spent In This Region")
        In_df2 = In_df.loc[In_df.index >= tstart]
        In_df2 = In_df2.div(In_df2[all_regions[region]], axis=0)

        # prevalence
        In_df2.plot(ax=axs.flat[0], color=[colordict[x] for x in In_df.columns])
        # betas
        beta_df[region].loc[beta_df.index >= tstart].plot(ax=axs.flat[1], color=(0.1, 0.1, 0.1))
        betap_df[region].loc[betap_df.index >= tstart].plot(ax=axs.flat[1], color='b', style='--')
        betal_df[region].loc[betal_df.index >= tstart].plot(ax=axs.flat[1], color='r', style='--')
        axs.flat[1].legend(['Disconnected', 'Pop Attached', 'Loc Attached'])
        # contact with other regions
        region_contact = []
        for j, date in enumerate(In_df.index):
            cm_j = cm['dwell_matrices'][mobility_dates[np.where(mobility_dates <= date)[0][-1]]]
            region_contact.append(np.dot(cm_j['dwell_rownorm'], np.transpose(cm_j['dwell_colnorm']))[i, :])
        region_contact = pd.DataFrame(np.stack(region_contact), index=In_df.index, columns=In_df.columns)
        rc2 = region_contact.drop(all_regions[region], axis=1).loc[region_contact.index >= tstart]
        rc2.plot(ax=axs.flat[2], color=[colordict[x] for x in rc2.columns])
        # Plot the Ds
        Dtildes = []
        Dstars = []
        for j, date in enumerate(In_df.index):
            cm_j = cm['dwell_matrices'][mobility_dates[np.where(mobility_dates <= date)[0][-1]]]
            Dtildes.append(cm_j['dwell_rownorm'][i, :])
            Dstars.append(cm_j['dwell_colnorm'][i, :])
        Dtildes = pd.DataFrame(np.stack(Dtildes), index=In_df.index, columns=In_df.columns)
        Dstars = pd.DataFrame(np.stack(Dstars), index=In_df.index, columns=In_df.columns)
        Dt2 = Dtildes.drop(all_regions[region], axis=1).loc[Dtildes.index >= tstart]
        Ds2 = Dstars.drop(all_regions[region], axis=1).loc[Dstars.index >= tstart]
        Dt2.plot(ax=axs.flat[3], color=[colordict[x] for x in Dt2.columns])
        Ds2.plot(ax=axs.flat[4], color=[colordict[x] for x in Ds2.columns])
        # save
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("output/connected_regional_model/" + name_prefix + "_mobility_detailed")
        plt.close()

        # plot disconnected vs Option 1 only
        f = plt.figure(figsize=(15, 8), dpi=300)
        ax = plt.gca()
        plt.title("Mobility Adjustments, Population Attached Beta: " + all_regions[region])
        beta_df[region].loc[beta_df.index >= tstart].plot(ax=ax, color=(0.1, 0.1, 0.1))
        betap_df[region].loc[betap_df.index >= tstart].plot(ax=ax, color='b', style='--')
        ax.legend(['Disconnected', 'Pop Attached'])
        f.tight_layout()
        plt.savefig("output/connected_regional_model/" + name_prefix + "_mobility_option1")
        plt.close()

        # plot disconnected vs Option 1 detailed
        f, axs = plt.subplots(3, 1, figsize=(15, 22), dpi=300)
        f.suptitle("Mobility Adjustments, Population Attached Beta: " + all_regions[region])
        axs.flat[0].set_title("This Region's Contact With Other Regions (Anywhere)")
        axs.flat[1].set_title("Relative Prevalence To This Region")
        axs.flat[2].set_title("Beta")
        rc2.plot(ax=axs.flat[0], color=[colordict[x] for x in rc2.columns])
        In_df2.plot(ax=axs.flat[1], color=[colordict[x] for x in In_df.columns])
        beta_df[region].loc[beta_df.index >= tstart].plot(ax=axs.flat[2], color=(0.1, 0.1, 0.1))
        betap_df[region].loc[betap_df.index >= tstart].plot(ax=axs.flat[2], color='b', style='--')
        axs[2].legend(['Disconnected', 'Pop Attached'])
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("output/connected_regional_model/" + name_prefix + "_mobility_option1_detailed")
        plt.close()

        # plot disconnected vs Option 2 only
        f = plt.figure(figsize=(15, 8), dpi=300)
        ax = plt.gca()
        plt.title("Mobility Adjustments, Location Attached Beta: " + all_regions[region])
        beta_df[region].loc[beta_df.index >= tstart].plot(ax=ax, color=(0.1, 0.1, 0.1))
        betal_df[region].loc[betal_df.index >= tstart].plot(ax=ax, color='r', style='--')
        ax.legend(['Disconnected', 'Loc Attached'])
        f.tight_layout()
        plt.savefig("output/connected_regional_model/" + name_prefix + "_mobility_option2")
        
        # plot disconnected vs Option 2 detailed
        f, axs = plt.subplots(4, 1, figsize=(15, 23), dpi=300)
        f.suptitle("Mobility Adjustments, Location Attached Beta: " + all_regions[region])
        axs.flat[0].set_title("Fraction of Time Spent In This Region Done By People From Other Regions")
        axs.flat[1].set_title("Fraction of Other Region's Time Spent In This Region")
        axs.flat[2].set_title("Relative Prevalence To This Region")
        axs.flat[3].set_title("Beta")
        Dt2.plot(ax=axs.flat[0], color=[colordict[x] for x in Dt2.columns])
        Ds2.plot(ax=axs.flat[1], color=[colordict[x] for x in Ds2.columns])
        In_df2.plot(ax=axs.flat[2], color=[colordict[x] for x in In_df.columns])
        betal_df[region].loc[betal_df.index >= tstart].plot(ax=axs.flat[3], color='r', style='--')
        beta_df[region].loc[beta_df.index >= tstart].plot(ax=axs.flat[3], color=(0.1, 0.1, 0.1))
        axs[3].legend(['Disconnected', 'Loc Attached'])
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("output/connected_regional_model/" + name_prefix + "_mobility_option2_detailed")
        plt.close()


if __name__ == '__main__':
    run()
