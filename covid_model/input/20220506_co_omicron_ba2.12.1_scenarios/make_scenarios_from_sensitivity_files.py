### Python Standard Library ###
import json
from itertools import product
### Third Party Imports ###
### Local Imports ###


def main():
    ams = json.load(open("sensitivity_attribute_multipliers.json"))

    am_scenarios = {}
    for label, sens_group in ams.items():
        labs = ["_".join(map(str, combo)) for combo in list(product(*[range(len(v)) for v in sens_group.values()]))]
        list(product(*list(sens_group.values())))
        for i, am in enumerate(product(*list(sens_group.values()))):
            am_scenarios[f'{label}_{labs[i]}'] = [ami for amlist in am for ami in amlist]
    s = "{\n"
    for scen, specs in am_scenarios.items():
        s += f'"{scen}": {json.dumps(specs)},\n'
    s += '}'
    with open('attribute_multipliers_scenarios.json', 'w') as f:
        f.write(s)


if __name__ == "__main__":
    main()