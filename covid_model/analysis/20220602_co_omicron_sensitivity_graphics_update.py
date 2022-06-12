### Python Standard Library ###
import os
import datetime as dt
import shutil
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
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
    results_dir = "20220506_co_omicron_ba2.12.1_scenarios.py"

    with open(get_filepath_prefix(outdir) + "________________________.txt", 'w') as f:
        f.write("")

    ####################################################################################################################
    # Run

    # load files
    df = pd.read_csv(f"{os.path.join('covid_model', 'output', results_dir, '')}20220507_043316_run_model_scenarios_hospitalized2_.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = pd.concat([df, df.series.str.rsplit("_", n=3, expand=True)], axis=1)

    dfnew = pd.read_csv(f"{os.path.join('covid_model', 'output', results_dir, '')}20220601_222750_run_model_scenarios_hospitalized2_.csv")
    dfnew['date'] = pd.to_datetime(dfnew['date'])
    dfnew = pd.concat([dfnew, dfnew.series.str.rsplit("_", n=3, expand=True)], axis=1)

    dfo = dfnew[dfnew['series'] == 'observed']
    df5 = df[df[0] == '5_percent']
    df5new = dfnew[dfnew[0] == '5_percent']

    # plot the projection along with the uncertainty in x

    from_date = dt.datetime.strptime("2021-12-01", "%Y-%m-%d")
    to_date = dt.datetime.strptime("2022-09-01", "%Y-%m-%d")
    dfo2 = dfo[(dfo.date >= from_date) & (dfo.date <= to_date)]

    df5 = df5[(df5.date >= from_date) & (df5.date <= to_date)]
    df5new = df5new[(df5new.date >= from_date) & (df5new.date <= to_date)]

    f = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.plot(dfo2.date, dfo2.hospitalized, color="black", label="Observed")
    for j, high_low, col in zip(['0', '1'], ['High', 'Low'], ['darkorange', 'blue']):
        dfp2_sens = df5[(df5[1] == j) & ~(df5[[2, 3]] == ['0', '0']).all(axis=1)]
        dfp2_med = df5[(df5[[1, 2, 3]] == [j, '0', '0']).all(axis=1)]
        dfp2_med.to_csv(get_filepath_prefix(outdir) + fname_prefix + f"_hospitalized_5_{high_low}.csv")
        dfp2_sens.to_csv(get_filepath_prefix(outdir) + fname_prefix + f"_hospitalized_5_{high_low}_sensitivity.csv")
        ax.plot(dfp2_med.date, dfp2_med.hospitalized, color=col, label=f"{high_low} BA.2.12.1 Immune Escape (Gov Briefing)", lw=2.0, alpha=0.5, linestyle='dashed')

        dfp2_sens = df5new[(df5new[1] == j) & ~(df5new[[2, 3]] == ['0', '0']).all(axis=1)]
        dfp2_med = df5new[(df5new[[1, 2, 3]] == [j, '0', '0']).all(axis=1)]
        for series in dfp2_sens.series.unique():
            ax.plot(df5new[df5new.series == series].date, df5new[df5new.series == series].hospitalized, alpha=0.3, color=col, lw=0.8)
        dfp2_med.to_csv(get_filepath_prefix(outdir) + fname_prefix + f"_hospitalized_5_{high_low}.csv")
        dfp2_sens.to_csv(get_filepath_prefix(outdir) + fname_prefix + f"_hospitalized_5_{high_low}_sensitivity.csv")
        ax.plot(dfp2_med.date, dfp2_med.hospitalized, color=col, label=f"{high_low} BA.2.12.1 Immune Escape (Updated)", lw=2.0)

    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.title(f'BA.2.12.1 Projections')
    plt.ylabel("Hospitalizations")
    plt.xlabel("Date")
    plt.legend(loc='upper right')
    plt.savefig(get_filepath_prefix(outdir) + fname_prefix + f"_hospitalized_5.png", dpi=300)

    # copy over legacy output files and the correct omicron reports
    df_legacy = pd.read_csv(os.path.join('covid_model', 'output', results_dir, "20220507_043316_out2_.csv"))
    df_legacy['date'] = pd.to_datetime(df_legacy['date'])
    df_legacy[(df_legacy.scen == "5_percent_0_0_0") & (df_legacy.date <= to_date)].to_csv(get_filepath_prefix(outdir) + fname_prefix + "_out2_high.csv")
    df_legacy[(df_legacy.scen == "5_percent_1_0_0") & (df_legacy.date <= to_date)].to_csv(get_filepath_prefix(outdir) + fname_prefix + "_out2_low.csv")
    # new version
    df_legacy2 = pd.read_csv(os.path.join('covid_model', 'output', results_dir, "20220601_222750_out2_.csv"))
    df_legacy2['date'] = pd.to_datetime(df_legacy2['date'])
    df_legacy2[(df_legacy2.scen == "5_percent_0_0_0") & (df_legacy2.date <= to_date)].to_csv(get_filepath_prefix(outdir) + fname_prefix + "_out2_high_new.csv")
    df_legacy2[(df_legacy2.scen == "5_percent_1_0_0") & (df_legacy2.date <= to_date)].to_csv(get_filepath_prefix(outdir) + fname_prefix + "_out2_low_new.csv")

    shutil.copy2(os.path.join('covid_model', 'output', results_dir, '20220507_021659_omicron_report_5_percent_0_0_0.png'), get_filepath_prefix(outdir) + '_omicron_report_high.png')
    shutil.copy2(os.path.join('covid_model', 'output', results_dir, '20220507_021118_omicron_report_5_percent_1_0_0.png'), get_filepath_prefix(outdir) + '_omicron_report_low.png')
    shutil.copy2(os.path.join('covid_model', 'output', results_dir, '20220601_172312_omicron_report_5_percent_0_0_0.png'), get_filepath_prefix(outdir) + '_omicron_report_high_new.png')
    shutil.copy2(os.path.join('covid_model', 'output', results_dir, '20220601_174927_omicron_report_5_percent_1_0_0.png'), get_filepath_prefix(outdir) + '_omicron_report_low_new.png')




    print("done")


if __name__ == "__main__":
    main()
