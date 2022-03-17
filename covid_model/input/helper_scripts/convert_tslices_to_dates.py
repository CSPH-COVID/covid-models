import json
from datetime import datetime as dt, timedelta as td


def convert(file_name):
    # because of the recursive nature of params files, let's just do a brute force approach to replacement.
    t0 = dt.strptime("2020-01-24", "%Y-%m-%d")
    with open(file_name, 'r') as f:
        fs = f.read()
    tslices_loc = fs.find("tslices")
    b2 = 0
    fs_new = []
    while tslices_loc > 0:
        b1 = fs.find("[", tslices_loc) + 1
        fs_new.append(fs[b2:b1])
        b2 = fs.find("]", b1)
        nums = [int(num) for num in fs[b1:b2].split(",")]
        dates = ", ".join(["\"" + dt.strftime(t0 + td(days=num), "%Y-%m-%d") + "\"" for num in nums])
        fs_new.append(dates)
        tslices_loc = fs.find("tslices", b2)
    fs_new.append(fs[b2:])
    fs_new = "".join(fs_new)
    with open("date_based_" + file_name, 'w') as f:
        f.write(fs_new)


if __name__ == "__main__":
    convert("params.json")
    convert("region_params.json")
    convert("vacc_proj_params.json")
