####################################################################################################################
### Saving Writing Data

def prepare_write_specs_query(self, tags=None):
    # returns all the data you would need to write to the database but doesn't actually write to the database
    if tags is not None:
        self.tags.update(tags)

    write_info = OrderedDict([
        ("created_at", dt.datetime.now()),
        ("base_spec_id", int(self.base_spec_id) if self.base_spec_id is not None else None),
        ("tags", json.dumps(self.tags)),
        ("start_date", self.start_date),
        ("end_date", self.end_date),
        ("tslices", self.tslices),
        ("tc", self.tc),
        ("tc_cov", json.dumps(self.tc_cov.tolist() if isinstance(self.tc_cov,
                                                                 np.ndarray) else self.tc_cov) if self.tc_cov is not None else None),
        ("model_params", json.dumps(self.model_params)),
        ("vacc_actual", json.dumps({dose: {";".join(key): val for key, val in
                                           rates.unstack(level=['region', 'age']).to_dict(orient='list').items()}
                                    for dose, rates in self.actual_vacc_df.to_dict(orient='series').items()})),
        ("vacc_proj_params", json.dumps(self.vacc_proj_params)),
        ("vacc_proj", json.dumps({dose: {";".join(key): val for key, val in
                                         rates.unstack(level=['region', 'age']).to_dict(orient='list').items()} for
                                  dose, rates in self.proj_vacc_df.to_dict(
                orient='series').items()} if self.proj_vacc_df is not None else None)),
        ("timeseries_effects", json.dumps(self.timeseries_effects)),
        ("attribute_multipliers", json.dumps(self.attribute_multipliers)),
        ("region_definitions", json.dumps(self.model_region_definitions)),
        ("mobility_actual", json.dumps(self.actual_mobility)),
        ("mobility_proj_params", json.dumps(self.mobility_proj_params)),
        ("mobility_proj", json.dumps(self.proj_mobility)),
        ("mobility_mode", self.model_mobility_mode),
        ("regions", json.dumps(self.regions))
    ])

    return write_info


@classmethod
def write_prepared_specs_to_db(cls, write_info, engine, spec_id: int = None):
    # writes the given info to the db without needing an explicit instance
    specs_table = get_sqa_table(engine, schema='covid_model', table='specifications')

    with Session(engine) as session:
        if spec_id is None:
            max_spec_id = session.query(func.max(specs_table.c.spec_id)).scalar()
            spec_id = max_spec_id + 1

        stmt = specs_table.insert().values(
            spec_id=spec_id,
            **write_info
        )
        session.execute(stmt)
        session.commit()
    return {**write_info, 'spec_id': spec_id}


def write_specs_to_db(self, engine, tags=None):
    # get write info
    write_info = self.prepare_write_specs_query(tags=tags)

    specs_table = get_sqa_table(engine, schema='covid_model', table='specifications')
    with Session(engine) as session:
        # generate a spec_id so we can assign it to ourselves
        max_spec_id = session.query(func.max(specs_table.c.spec_id)).scalar()
        self.spec_id = max_spec_id + 1
    return self.write_prepared_specs_to_db(write_info, engine, spec_id=self.spec_id)


def prepare_write_results_query(self, vals_json_attr='seir', cmpts_json_attrs=('region', 'age', 'vacc'), sim=False):
    # build data frame with index of (t, region, age, vacc) and one column per seir cmpt
    solution_sum_df = self.solution_sum([vals_json_attr] + list(cmpts_json_attrs)).stack(cmpts_json_attrs)

    # build export dataframe
    df = pd.DataFrame(index=solution_sum_df.index)
    df['t'] = solution_sum_df.index.get_level_values('t')
    df['cmpt'] = solution_sum_df.index.droplevel('t').to_frame().to_dict(
        orient='records') if solution_sum_df.index.nlevels > 1 else None
    df['vals'] = solution_sum_df.to_dict(orient='records')
    for col in ['cmpt', 'vals']:
        df[col] = df[col].map(lambda d: json.dumps(d, ensure_ascii=False))

    # if a sim_id is provided, insert it as a simulation result; some fields are different
    if not sim:
        # build unique parameters dataframe
        params_df = self.params_as_df
        grouped = params_df.groupby(['t'] + list(cmpts_json_attrs))
        unique_params = [param for param, is_unique in (grouped.nunique() == 1).all().iteritems() if is_unique]
        unique_params_df = grouped.max()[unique_params]
        df['params'] = unique_params_df.apply(lambda x: json.dumps(x.to_dict(), ensure_ascii=False), axis=1)
    else:
        df['tc'] = json.dumps(self.tc)
    return df


@classmethod
def write_prepared_results_to_db(cls, df, engine, spec_id, sim_id=None, sim_result_id=None):
    if sim_id is None:
        table = 'results_v2'
        df['created_at'] = dt.datetime.now()
        df['spec_id'] = spec_id
        df['result_id'] = \
            pd.read_sql(f'select coalesce(max(result_id), 0) from covid_model.{table}', con=engine).values[0][0] + 1
    else:
        df['sim_id'] = sim_id
        df['sim_result_id'] = sim_result_id
        table = 'simulation_results_v2'

    # write to database
    chunksize = int(np.floor(9000.0 / df.shape[
        1]))  # max parameters is 10,000. Assume 1 param per column and give some wiggle room because 10,000 doesn't always work
    results = df.to_sql(table
                        , con=engine, schema='covid_model'
                        , index=False, if_exists='append', method='multi', chunksize=chunksize)
    return df


# write to covid_model.results
def write_results_to_db(self, engine, new_spec=False, vals_json_attr='seir',
                        cmpts_json_attrs=('region', 'age', 'vacc'), sim_id=None, sim_result_id=None):
    write_df = self.prepare_write_results_query(vals_json_attr, cmpts_json_attrs, sim=sim_id is not None)

    # if there's no existing spec_id assigned, write specs to db to get one
    if self.spec_id is None or new_spec:
        self.write_specs_to_db(engine)

    df = CovidModel.write_prepared_results_to_db(write_df, engine, self.spec_id, sim_id, sim_result_id)
    self.result_id = df['result_id'][0][0]
    return df