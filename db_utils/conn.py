### Python Standard Library ###
import os
import json
### Third Party Imports ###
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
### Local Imports ###


def db_engine(db_name_env=None, db_user_env=None, db_host_env=None, db_credentials_env=None, env_prefix=''):
    db_name = os.environ[db_name_env if db_name_env is not None else (env_prefix+'db_name')]
    db_user = os.environ[db_user_env if db_user_env is not None else (env_prefix+'db_user')]
    db_host = os.environ[db_host_env if db_host_env is not None else (env_prefix+'db_host')]
    db_credentials = os.environ[db_credentials_env if db_credentials_env is not None else (env_prefix+'db_credentials')]

    conn_string = f'postgresql://{db_user}:{db_credentials}@{db_host}:5432/{db_name}'

    return create_engine(conn_string)


def db_connect(db_name_env=None, db_user_env=None, db_host_env=None, db_credentials_env=None, env_prefix=''):
    db_name = os.environ[db_name_env if db_name_env is not None else (env_prefix+'db_name')]
    db_user = os.environ[db_user_env if db_user_env is not None else (env_prefix+'db_user')]
    db_host = os.environ[db_host_env if db_host_env is not None else (env_prefix+'db_host')]
    db_credentials = os.environ[db_credentials_env if db_credentials_env is not None else (env_prefix+'db_credentials')]

    conn_string = "dbname='" + str(db_name) + "' user='" + str(db_user) + "' host='" + str(db_host) + "' password='" + str(db_credentials) + "'"

    try:
        conn = psycopg2.connect(str(conn_string))
        conn.autocommit = True
        return conn
    except:
        print("Unable to connect to the database")
        raise


def fetch_data(engine, sql=None, sql_path=None, save_to_file=None, force_rerun=True, manifest_path=None, update_manifest=True, **pd_read_args):
    # get manifest file
    manifest = None
    this_manifest = None
    if manifest_path:
        manifest_file = open(manifest_path, 'r')
        manifest = json.load(manifest_file) if os.stat(manifest_path).st_size != 0 else []
        this_manifest_list = [m for m in manifest if m['csv_file'] == save_to_file]
        this_manifest = this_manifest_list[0] if len(this_manifest_list) > 0 else None
    # get data from csv, if appropriate
    if save_to_file and not force_rerun and os.path.exists(save_to_file):
        if this_manifest:
            pd_read_args['parse_dates'] = [k for k, v in this_manifest['dtypes'].items() if v == 'datetime64[ns]']
            pd_read_args['dtype'] = {k: v for k, v in this_manifest['dtypes'].items() if v != 'datetime64[ns]'}
        data = pd.read_csv(save_to_file, **pd_read_args)
    # otherwise get data from db
    else:
        # get sql from sql_path, if appropriate
        if sql_path is not None:
            if sql is None:
                with open(sql_path) as sql_file:
                    sql = sql_file.read()
            else:
                raise ValueError("Either `sql` or `sql_path` must be None.")
        # get data from db
        data = pd.read_sql(sql, engine, **pd_read_args)
        if save_to_file:
            data.to_csv(save_to_file, index=False)
        # update manifest, if appropriate
        if manifest is not None and update_manifest:
            manifest_file = open(manifest_path, 'w')
            if not this_manifest:
                this_manifest = {'sql': sql_path if sql_path else sql, 'csv_file': save_to_file}
                manifest.append(this_manifest)
            this_manifest['dtypes'] = {k: str(v) for k, v in list(data.index.to_frame().dtypes.to_dict().items()) + list(data.dtypes.to_dict().items())}
            json.dump(manifest, manifest_file, indent=4)

    return data
