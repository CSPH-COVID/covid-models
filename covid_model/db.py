import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
import json


def db_engine(db_name_env=None, db_user_env=None, db_host_env=None, db_credentials_env=None, env_prefix='', db_type='bigquery'):
    if db_type == 'postgres':
        db_name = os.environ[db_name_env if db_name_env is not None else (env_prefix+'db_name')]
        db_user = os.environ[db_user_env if db_user_env is not None else (env_prefix+'db_user')]
        db_host = os.environ[db_host_env if db_host_env is not None else (env_prefix+'db_host')]
        db_credentials = os.environ[db_credentials_env if db_credentials_env is not None else (env_prefix+'db_credentials')]
        conn_string = f'postgresql://{db_user}:{db_credentials}@{db_host}:5432/{db_name}'
    elif db_type == 'bigquery':
        conn_string = f'bigquery://{os.environ["gcp_project"]}'
    else:
        raise ValueError('db_type must be "postgres" or "bigquery"')

    return create_engine(conn_string)


def get_sqa_table(engine, schema, table):
    if str(engine.engine.url).split('://')[0] == 'bigquery':
        return Table(f'{schema}.{table}', MetaData(bind=engine), autoload=True)
    else:
        metadata = MetaData(schema=schema)
        metadata.reflect(engine, only=['specifications'])
        return metadata.tables[f'{schema}.{table}']
