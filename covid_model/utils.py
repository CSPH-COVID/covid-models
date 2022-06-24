### Python Standard Library ###
from os.path import join
import datetime as dt
import logging
import traceback
import os
### Third Party Imports ###
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
### Local Imports ###


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

# tslices and values define a stepwise function; get the value of that function for a given t
def get_value_from_slices(tslices, values, t):
    if len(values) != len(tslices) - 1:
        raise ValueError(f"Length of values ({len(values)}) must equal length of tslices ({len(tslices)}) - 1.")
    for i in range(len(tslices)-1):
        if tslices[i] <= t < tslices[i+1]:
            # if i > len(values):
            #     raise ValueError(f"")
            return values[i]
    raise ValueError(f"Cannot fetch value from slices because t={t} is out of range.")


# recursive function to process parameters that include values for different time slices, and construct params for a specific t
def get_params(input_params, t, tslices=None):
    if type(input_params) == list:
        value = get_value_from_slices([0]+tslices+[99999], input_params, t)
        return get_params(value, t, tslices)
    elif type(input_params) == dict:
        if 'tslices' in input_params.keys():
            return get_params(input_params['value'], t, tslices=input_params['tslices'])
        else:
            return {k: get_params(v, t, tslices) for k, v in input_params.items()}
    else:
        return input_params


# Formatter class which indents based on how deep in the frame stack we are.
class IndentLogger(logging.LoggerAdapter):
    """
    use this adapter with:
        import logging
        logger = IndentLogger(logging.getLogger(''), {})
        logger.info(...)
    """
    @staticmethod
    def indent():
        indentation_level = len(traceback.extract_stack())
        return indentation_level-5

    def process(self, msg, kwargs):
        return "{}{}".format('-' * self.indent() + "|", msg), kwargs


def setup(name, log_level="info"):
    outdir = os.path.join("covid_model", "output", name)
    os.makedirs(outdir, exist_ok=True)

    # parse arguments
    log_dict = {'debug': logging.DEBUG,
                'info': logging.INFO,
                'warning': logging.WARNING,
                'error': logging.ERROR,
                'critical': logging.CRITICAL}

    #log_choices = ['debug', 'info', 'warning', 'error', 'critical']
    #parser.add_argument('-l', '--log', nargs='?', choices=log_choices, default='info', const='info', help="logging level for console")

    # set up logging to file
    # set up logging to file (file always does debug)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-4s|%(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        filename=get_filepath_prefix(outdir) + name + "________________________________________.log",
                        filemode='a')
    # set up logging to console (only do this if there are no handlers)
    if len(logging.getLogger('').handlers) < 2:
        console = logging.StreamHandler()
        console.setLevel(log_dict[log_level])
        formatter = logging.Formatter('%(asctime)s %(levelname)-4s|%(message)s', '%Y/%m/%d %H:%M:%S')
        console.formatter = formatter
        logging.getLogger('').addHandler(console)
    logging.info("============================================================")

    # stop the incessant warning messages about missing fonts from matplotlib
    logging.getLogger('matplotlib.font_manager').disabled = True

    return outdir


def get_filepath_prefix(outdir = None, tags=None):
    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = join(outdir, now) if outdir is not None else now
    if tags:
        prefix += f'_{"_".join(str(key) + "_" + str(val) for key, val in tags.items())}'
    return prefix + '_'