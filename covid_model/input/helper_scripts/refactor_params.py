import json


def main():
    with open('params.json', 'r') as f:
        params = json.loads(f.read())
    with open('region_params.json', 'r') as f:
        region_params = json.loads(f.read())

    def process_params_dict(params_dict, region="co"):
        new_dict = {}
        for key, val in params_dict.items():
            if type(val) is not dict:
                param = [{"tslices": None, "attributes": {"region":region}, "values": val}]
            elif 'tslices' not in val.keys():
                param = []
                for age_key, age_val in val.items():
                    param.append({"tslices": None, "attributes": {"region":region, "age":age_key}, "values": age_val})
            else:
                if type(val['value']) is dict:
                    param = []
                    for age_key, age_val in val['value'].items():
                        param.append({"tslices": val['tslices'], "attributes": {"region": region, "age":age_key}, "values": age_val})
                else:
                    param = [{"tslices": val['tslices'], "attributes": {"region": region}, "values": val['value']}]
            new_dict[key] = param
        return(new_dict)


    new_params = process_params_dict(params)
    new_region_params = [process_params_dict(rval, rkey) for rkey, rval in region_params.items()]

    # combine params and region params
    for rps in new_region_params:
        for param_name, param_list in rps.items():
            if param_name in new_params.keys():
                new_params[param_name].extend(param_list)
            else:
                new_params[param_name] = param_list

    # if "co" is the only region attribute represented in a parameter, remove the region attribute
    for param_name in new_params.keys():
        regions = set([param_dict['attributes']['region'] if 'region' in param_dict['attributes'] else None for param_dict in new_params[param_name]])
        if len(regions) == 1 and "co" in regions:
            for i in range(len(new_params[param_name])):
                del new_params[param_name][i]['attributes']['region']

    if (False):
        # easy but ugly
        with open("new_params.json", 'w') as f:
            json.dump(new_params, f, indent=2)
    else:
        # harder but prettier
        with open("new_params.json", 'w') as f:
            f.write("{\n")
            for param_name, param_list in new_params.items():
                f.write(f'  "{param_name}": [\n')
                for l in param_list:
                    tslices_string = "[" + ", ".join("\"" + str(ts) + "\"" for ts in l["tslices"]) + "]" if l['tslices'] is not None else "null"
                    attributes_string = json.dumps(l['attributes'])
                    values_string = "[" + ", ".join(str(v) for v in l["values"]) + "]" if l['tslices'] is not None else l['values']
                    f.write(f'    {{"tslices": {tslices_string}, "attributes": {attributes_string}, "values": {values_string}}},\n')
                f.write("   ],\n")
            f.write("}\n")
        print("done")


if __name__ == "__main__":
    main()
