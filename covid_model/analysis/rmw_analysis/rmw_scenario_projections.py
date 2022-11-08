#!/usr/bin/env python
# coding: utf-8

# ### Setup

# In[4]:


""" Python Standard Library """
import os
import datetime as dt
import json
import logging
import pickle
""" Third Party Imports """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
""" Local Imports """
if 'requirements.txt' not in os.listdir(os.getcwd()):
    os.chdir(os.path.join('../../../../..', '..', '..'))
print(os.getcwd())
# Import the RMW model instead of the original model
from covid_model.rmw_model import CovidModel as RMWCovidModel
from covid_model.runnable_functions import do_regions_fit, do_single_fit, do_fit_scenarios, do_create_multiple_reports
from covid_model.utils import setup, get_filepath_prefix
from covid_model.analysis.charts import plot_transmission_control

# os.environ['gcp_project'] = 'co-covid-models'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "co-covid-models-credentials.json"


# In[5]:


# set up the output directory for this Jupyter notebook
outdir = setup("rmw_scenario_projections.ipynb")


# ### Fit an initial scenario through February 2022

# In[6]:


# designate the arguments for how the model will behave
model_args = {
    'params_defs': 'covid_model/input/rmw_params.json',
    'region_defs': 'covid_model/input/rmw_region_definitions.json',
    'vacc_proj_params': 'covid_model/analysis/20221004_oct_gov_briefing/20221004_vacc_proj_params.json',
    'start_date': '2020-01-24',
    'end_date': '2024-01-01',
    'regions': ['coe', 'con', 'cow']
}

# this is how the fit will behave
# place the outdir argument here to tell the model fit where to go
fit_args = {'outdir': outdir,
            'fit_end_date': '2022-02-28',
            'model_class':RMWCovidModel
}

# because all the scenarios are the same
# List of regions to use
regions_to_use = ["coe","cow"]
scen_args = [{"regions":[region]} for region in regions_to_use]
#model = do_single_fit(**model_args,**fit_args)
models = do_fit_scenarios(base_model_args=model_args,
                         fit_args=fit_args,
                         scenario_args_list=scen_args)

for reg,model in zip(scen_args,models):
    with open(f"solution_df_{reg}.pkl","wb") as f:
        pickle.dump(model.solution_ydf,f)

exit(0)
# ### Create and run scenarios from Feb 2022 to present

# In[ ]:


multiprocess = 4

scenario_params = json.load(open("covid_model/analysis/20221004_oct_gov_briefing/20221004_scenario_params.json"))

model_args = {
    'base_spec_id': 4167 #model.spec_id, # use the spec id that was output from the model fit
}
model_fit_args = {
    'outdir': outdir,
    'fit_start_date': '2022-03-01', # set the start date for the earliest point at which the scenarios start to differ from one another
    'pre_solve_model': True # force the model to establish initial conditions so the fit can start on the fit start date
}

# define vaccine effectiveness for < 5 (this is a multiplier for the baseline vaccine effectiveness for 0-19)
vacc_eff_lt5 = 0.5

# Create different scenarios to model
scenario_model_args = []
for vx_seed in [0, 5]:
    for vir_mult in [0.833, 2.38]:
        hrf = {"2020-01-01": 1, "2022-03-01": (0.66 + 0.34*0.8),
               "2022-03-15": (0.34 + 0.66*0.8), "2022-03-30": 0.8}
        vx_adjust = [{"param": "vx_seed",
                      "vals": {"2020-01-01": 0, "2022-09-30": vx_seed, "2022-10-30": 0},
                      "desc": "Variant X seeding"}]
        vir_adjust = [{"param": "hosp",
                       "attrs": {"variant": "vx"},
                       "mults": {"2020-01-01": vir_mult},
                       "desc": "Variant X hospitalization multiplier"}]
        lt5_vacc_adjust = [{"param": "immunity",
                            "attrs": {'age': '0-19', 'vacc': 'shot1'},
                            "mults": {"2020-01-01": 1,
                                      "2022-06-20": 0.99 + 0.01*vacc_eff_lt5,
                                      "2022-06-30": 0.98 + 0.02*vacc_eff_lt5,
                                      "2022-07-10": 0.97 + 0.03*vacc_eff_lt5,
                                      "2022-07-20": 0.96 + 0.04*vacc_eff_lt5,
                                      "2022-08-10": 0.95 + 0.05*vacc_eff_lt5,
                                      "2022-08-30": 0.94 + 0.06*vacc_eff_lt5,
                                      "2022-09-20": 0.93 + 0.07*vacc_eff_lt5},
                            "desc": "weighted average using share of 0-19 getting shot1 who are under 5"}]
        scenario_model_args.append({'params_defs': scenario_params + vx_adjust + vir_adjust + lt5_vacc_adjust,
                                    'hosp_reporting_frac': hrf,
                                    'tags': {'vx_seed': vx_seed,
                                             'vir_mult': vir_mult,
                                             'booster_mult': 0}})
            


# In[ ]:


# check how many scenarios there are
len(scenario_model_args)


# In[ ]:


# run the scenarios
models = do_fit_scenarios(base_model_args=model_args, scenario_args_list=scenario_model_args, fit_args=model_fit_args, multiprocess=multiprocess)


# ### Run the report for each fit model

# In[ ]:


# here you can also specify which variants you want to calculate immunity for
do_create_multiple_reports(models, multiprocess=multiprocess, outdir=outdir, prep_model=False, solve_model=True, immun_variants=['ba45', 'vx'], from_date='2022-01-01')


# In[ ]:


logging.info('Projecting')
for model in models:
    logging.info('')
    #model.prep()  # don't think we need to prep anymore.
    model.solve_seir()

    model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'states_seir_variant_immun_total_all_at_once_projection.csv')
    model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'states_full_projection.csv')

    logging.info(f'{str(model.tags)}: Running forward sim')
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(211)
    hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
    hosps_df.plot(ax=ax)
    ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(), dt.datetime.strptime('2024-01-01', "%Y-%m-%d").date())
    ax = fig.add_subplot(212)
    plot_transmission_control(model, ax=ax)
    ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(), dt.datetime.strptime('2024-01-01', "%Y-%m-%d").date())
    plt.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast.png')
    plt.close()
    hosps_df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + '_model_forecast.csv')
    json.dump(model.tc, open(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast_tc.json', 'w'))

logging.info('Running reports')


# ### Test Stuff

# In[ ]:


regions = {
"coe": {"name": "Colorado East", "counties": ["Baca", "Bent", "Cheyenne", "Crowley", "Custer", "El Paso", "Fremont", "Huerfano", "Kiowa", "Kit Carson", "Las Animas", "Lincoln", "Logan", "Morgan", "Otero", "Phillips", "Prowers", "Pueblo", "Sedgwick", "Teller", "Washington", "Yuma"], "counties_fips": ["8009", "8011", "8017", "8025", "8027", "8041", "8043", "8055", "8061", "8063", "8071", "8073", "8075", "8087", "8089", "8095", "8099", "8101", "8115", "8119", "8121", "8125"]},
"con": {"name": "Colorado North", "counties":["Adams", "Arapahoe", "Boulder", "Broomfield", "Clear Creek", "Denver", "Douglas", "Elbert", "Gilpin", "Grand", "Jackson", "Jefferson", "Larimer", "Park", "Summit", "Weld"], "counties_fips": [ "8001", "8005", "8013", "8014", "8019", "8031" ,"8035", "8039","8047","8049","8057","8059","8069","8093","8117","8123"]},
"cow": {"name": "Colorado West", "counties": ["Chaffee", "Conejos", "Costilla", "Delta", "Dolores", "Eagle", "Garfield", "Hinsdale", "La Plata", "Mesa", "Mineral", "Moffat", "Montezuma", "Montrose", "Ouray", "Pitkin", "Rio Blanco", "Rio Grande", "Routt", "Saguache", "San Juan", "San Miguel", "Alamosa", "Archuleta"], "counties_fips": ["8015", "8021", "8023","8029","8033","8037","8045","8051", "8053", "8067", "8065", "8077", "8079", "8081", "8083", "8085", "8091", "8097", "8103", "8105", "8107", "8109", "8111", "8113", "8003","8007"]},
"ide": {"name": "Idaho East", "counties": ["Bannock", "Bear Lake", "Benewah", "Bingham", "Bonneville", "Butte", "Caribou", "Clark", "Custer", "Franklin", "Fremont", "Jefferson", "Lemhi County", "Madison", "Oneida", "Power", "Teton"], "counties_fips": ["16005", "16007","16009","16011","16019","16023","16029","16033","16037","16041","16043","16051","16059","16065", "16071", "16077", "16081"]},
"idn": {"name": "Idaho North", "counties": ["Bonner","Boundary", "Clearwater","Idaho", "Kootenai", "Latah", "Lewis", "Nez Perce", "Shoshone"], "counties_fips": ["16017", "16021", "16035", "16049", "16055", "16057", "16061", "16069", "16079"]},
"ids": {"name": "Idaho South", "counties": ["Blaine", "Camas", "Cassia", "Gooding", "Jerome", "Lincoln", "Minidoka", "Twin Falls"], "counties_fips": ["16013", "16025", "16031", "16047", "16053", "16063", "16067", "16083"]},
"idw": {"name": "Idaho West", "counties": ["Ada", "Adams", "Boise", "Canyon", "Elmore", "Gem", "Owyhee", "Payette", "Valley", "Washington"], "counties_fips": ["16001", "16003", "16015", "16027", "16039", "16045", "16073", "16075", "16085", "16087"]},
"mte": {"name": "Montana East", "counties": ["Big Horn", "Carbon", "Carter", "Custer", "Daniels", "Dawson", "Fallon", "Fergus", "Garfield", "Golden Valley", "Judith Basin", "McCone", "Musselshell", "Petroleum", "Phillips", "Powder River", "Prairie", "Richland", "Roosevelt", "Rosebud", "Sheridan", "Stillwater", "Sweet Grass", "Treasure", "Valley", "Wheatland", "Wibaux", "Yellowstone"], "counties_fips": [ "30003", "30009", "30011", "30017", "30019", "30021", "30025", "30027", "30033", "30037", "30045", "30055", "30065", "30069", "30071", "30075", "30079", "30083", "30085", "30087", "30091", "30095", "30097", "30103", "30105", "30107", "30109", "30111"]},
"mtn": {"name": "Montana North", "counties": ["Blaine", "Cascade", "Chouteau", "Glacier", "Hill", "Liberty", "Pondera", "Teton", "Toole"], "counties_fips": ["30005", "30013", "30015", "30035", "30041", "30051", "30073", "30099", "30101"]},
"mtw": {"name": "Montana West", "counties": ["Beaverhead", "Broadwater", "Deer Lodge", "Flathead", "Gallatin", "Granite", "Jefferson", "Lake", "Lewis and Clark", "Lincoln", "Madison", "Meagher", "Mineral", "Missoula", "Park", "Powell", "Ravalli", "Sanders", "Silver Bow"], "counties_fips": ["30001", "30007", "30023", "30029", "30031", "30039", "30043", "30047", "30049", "30053", "30057", "30059", "30061", "30063", "30067", "30077", "30081", "30089", "30093"]},
"nme": {"name": "New Mexico East", "counties": ["Colfax", "Curry", "De Baca", "Guadalupe", "Harding", "Mora", "Quay", "Roosevelt", "San Miguel", "Union"], "counties_fips": ["35007", "", "35009", "35011", "35019", "35021", "35033", "35037", "35041", "35047", "35059"]},
"nmn": {"name": "New Mexico North", "counties": ["Los Alamos", "Rio Arriba", "Santa Fe", "Taos"], "counties_fips": ["35028", "35039", "35049", "35055"]},
"nms": {"name": "New Mexico South", "counties": ["Catron", "Chaves", "Dona Ana", "Eddy", "Grant", "Hidalgo", "Lea", "Lincoln", "Luna", "Otero", "Sierra"], "counties_fips": ["35003", "35005", "35013", "35015", "35017", "35023", "35025", "35027", "35029", "35035", "35051"]},
"nmw": {"name": "New Mexico West", "counties": ["Bernalillo", "Cibola", "McKinley", "San Juan", "Sandoval", "Socorro", "Torrance", "Valencia"], "counties_fips": ["35001", "35006", "35031", "35045", "35043", "35053", "35057", "35061"]},
"ute": {"name": "Utah East", "counties": ["Carbon", "Daggett", "Duchesne", "Emery", "Grand", "San Juan", "Uintah"], "counties_fips": ["49007", "49009", "49013", "49015", "49019", "49037", "49047"]},
"utn": {"name": "Utah North", "counties": ["Box Elder", "Cache", "Davis", "Morgan", "Rich", "Weber"], "counties_fips": ["49003", "49005", "49011", "49029", "49033", "49057"]},
"uts": {"name": "Utah South", "counties": ["Beaver", "Garfield", "Iron", "Kane", "Washington"], "counties_fips": ["49001", "49017", "49021", "49025", "49053"]},
"utw": {"name": "Utah West", "counties": ["Juab", "Millard", "Piute", "Salt Lake", "Sanpete", "Sevier", "Summit", "Tooele", "Utah", "Wasatch", "Wayne"], "counties_fips": ["49023", "49027", "49031", "49035", "49039", "49041", "49043", "49045", "49049", "49051", "49055"]},
"wye": {"name": "Wyoming East", "counties": ["Albany", "Carbon", "Converse", "Fremont", "Goshen", "Laramie", "Natrona", "Niobrara", "Platte"], "counties_fips": ["56001", "56007", "56009", "56013", "56015", "56021", "56025", "56027", "56031"]},
"wyn": {"name": "Wyoming North", "counties": ["Campbell", "Crook", "Johnson", "Sheridan", "Weston"], "counties_fips": ["56005", "56011", "56019", "56033", "56045"]},
"wyw": {"name": "Wyoming West", "counties": ["Big Horn", "Hot Springs", "Lincoln", "Park", "Sublette", "Sweetwater", "Teton", "Uinta", "Washakie"], "counties_fips": ["56003", "56017", "56023", "56029", "56035", "56037", "56039", "56041", "56043"]}
}


# In[ ]:


regions_picked = ["coe","con","cow"]


# In[ ]:


region_names = pd.DataFrame.from_dict({'region': [regions[region]["name"] for region in regions_picked]})

#region_names = pd.DataFrame.from_dict({'region': [fips for region in regions_picked for fips in regions[region]['counties_fips']]})




# In[ ]:


regions["coe"]["name"]


# In[ ]:


region_names


# In[ ]:




