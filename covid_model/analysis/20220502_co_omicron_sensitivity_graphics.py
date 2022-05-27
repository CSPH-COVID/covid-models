### Python Standard Library ###
import os
import datetime as dt
import json
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
### Local Imports ###
from covid_model.utils import get_filepath_prefix


def main():
    ####################################################################################################################
    # Set Up Arguments for Running
    basename = os.path.basename(__file__)
    fname_prefix = basename.split(".")[0]
    outdir = os.path.join("covid_model", "output", basename)
    os.makedirs(outdir, exist_ok=True)
    results_dir = "20220502_co_omicron_sensitivity.py"
    results_dir2 = "20220502_co_omicron_ba2.12.1_scenarios.py"

    with open(get_filepath_prefix(outdir) + "________________________.txt", 'w') as f:
        f.write("")

    ####################################################################################################################
    # Run

    # load files
    print('Loading hospitalizations')

    df = pd.read_csv(f"{os.path.join('covid_model', 'output', results_dir, '')}20220503_135015_run_model_scenarios_hospitalized2_.csv")
    df0 = pd.read_csv(f"{os.path.join('covid_model', 'output', results_dir2, '')}20220503_094658_None_run_solve_seir_hospitalized_BA.2.12.1 5x infectious as BA.2.csv")

    df['date'] = pd.to_datetime(df['date'])
    df0['date'] = pd.to_datetime(df0['date'])

    dfo = df[df['series']=='observed']
    df1 = df[df['series'].str.match('1\_')]
    df5 = df[df['series'].str.match('5')]
    df10 = df[df['series'].str.match('10')]

    # plot the projection along with the uncertainty in x

    from_date = dt.datetime.strptime("2021-12-01", "%Y-%m-%d")
    dfo2 = dfo[dfo.date > from_date]
    df02 = df0[df0.date > from_date]

    for dfp, sens in zip([df1, df5, df10], ["1", "5", "10"]):
        dfp2 = dfp[dfp.date > from_date]

        f = plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ax.plot(dfo2.date, dfo2.hospitalized, color="blue", label="Observed")
        first = True
        for series in dfp2['series'].unique():
            if first:
                ax.plot(dfp2[dfp2['series']==series].date, dfp2[dfp2['series'] == series].hospitalized, alpha=0.3, color="red", label="Sensitivity")
                first = False
            else:
                ax.plot(dfp2[dfp2['series'] == series].date, dfp2[dfp2['series'] == series].hospitalized, alpha=0.3, color="red")
        ax.plot(df02.date, df02.modeled_hospitalized, color="green", label="Model Projection")

        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.title(f'BA.2.12.1 Projection w/ {sens}% Sensitivity')
        plt.ylabel("Hospitalizations")
        plt.xlabel("Date")
        plt.legend(loc='upper right')
        plt.savefig(get_filepath_prefix(outdir) + fname_prefix + f"_hospitalized_{sens}.png", dpi=300)


    print("done")


if __name__ == "__main__":
    main()