### Python Standard Library ###
import json
from itertools import product
### Third Party Imports ###
### Local Imports ###


def main():
    ams = json.load(open("sensitivity_attribute_multipliers.json"))
    #ps = json.load(open("sensitivity_params.json"))

    am_scenarios = {}
    #ps_scenarios = {}
    for label, sens_group in ams.items():
        i = 1
        for am in product(*list(sens_group.values())):
            #for p in product(*list(ps.values())):
            am_scenarios[f'{label}_{i}'] = am
                #ps_scenarios[i] = p
            i += 1
    with open('attribute_multipliers_scenarios.json', 'w') as f:
        json.dump(am_scenarios, f, indent=2)
    #with open('params_scenarios.json', 'w') as f:
        #json.dump(ps_scenarios, f, indent=2)


if __name__ == "__main__":
    main()