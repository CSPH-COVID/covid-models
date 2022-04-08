### Python Standard Library ###
import json
from collections import OrderedDict
### Third Party Imports ###
### Local Imports ###


regions = OrderedDict([("cent", "Central"),
            ("cm", "Central Mountains"),
            ("met", "Metro"),
            ("ms", "Metro South"),
            ("ne", "Northeast"),
            ("nw", "Northwest"),
            ("slv", "San Luis Valley"),
            ("sc", "South Central"),
            ("sec", "Southeast Central"),
            ("sw", "Southwest"),
            ("wcp", "West Central Partnership")])


def get():
    # because of the recursive nature of params files, let's just do a brute force approach to replacement.
    with open('params.json', 'r') as f:
        js = json.load(f)
    p0 = js['total_pop']
    with open('region_params.json', 'r') as f:
        js = json.load(f)
    ps = [(r, js[r]["total_pop"], js[r]['total_pop']/p0) for r in js.keys() if r in regions.keys()]
    _ = [print(str(p)) for p in ps]

    for r, p, pr in ps:
        print('{"param": "initial_seed", "attrs": {"region": "' + r + '" }, "mult":' + str(pr) + ', "desc": "Based on ratio of region population to total population"},')
    for r, p, pr in ps:
        print('{"param": "alpha_seed", "attrs": {"region": "' + r + '" }, "mult":' + str(pr) + ', "desc": "Based on ratio of region population to total population"},')
    for r, p, pr in ps:
        print('{"param": "delta_seed", "attrs": {"region": "' + r + '" }, "mult":' + str(pr) + ', "desc": "Based on ratio of region population to total population"},')
    for r, p, pr in ps:
        print('{"param": "om_seed", "attrs": {"region": "' + r + '" }, "mult":' + str(pr) + ', "desc": "Based on ratio of region population to total population"},')


if __name__ == "__main__":
    get()
