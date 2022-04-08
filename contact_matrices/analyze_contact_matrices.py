### Python Standard Library ###
import os
from collections import defaultdict
from pprint import pprint
### Third Party Imports ###
import numpy as np
### Local Imports ###


if __name__ == '__main__':
    path = 'input/contact_matrices_raw'
    loc_cms = {}
    for cmf in os.listdir('input/contact_matrices_raw'):
        raw = np.loadtxt(os.path.join(path, cmf), delimiter=',')
        doubled = raw + raw.transpose()
        loc_cms[cmf[:-4]] = doubled
        # cms[cmf]

    loc_cat_mapping = {'School': 'school', 'Respondents workplace': 'work'}
    loc_cat_mapping.update({loc: 'other' for loc in loc_cms.keys() if loc not in loc_cat_mapping.keys()})

    cat_cms = defaultdict(float)
    for loc, cm in loc_cms.items():
        cat_cms[loc_cat_mapping[loc]] += cm
    pprint(dict(cat_cms))