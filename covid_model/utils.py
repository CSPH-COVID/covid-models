""" Python Standard Library """
from os.path import join
import datetime as dt
import logging
import traceback
import os

""" Third Party Imports """
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table

""" Local Imports """


def db_engine(db_name_env=None, db_user_env=None, db_host_env=None, db_credentials_env=None, env_prefix='',
              db_type='bigquery'):
    """Generic database connection, can be either PostGreSQL or Google BigQuery

    Args:
        db_name_env: database name to store in an environment variable (for postgres)
        db_user_env: user name to store in an environment variable (for postgres)
        db_host_env: host name to store in an environment variable (for postgres)
        db_credentials_env: credentials string to store in an environment variable (for postgres)
        env_prefix: prefix for all of the above variables (for postgres)
        db_type: either 'postgres' or 'bigquery'

    Returns: an active database connection

    """
    if db_type == 'postgres':
        db_name = os.environ[db_name_env if db_name_env is not None else (env_prefix + 'db_name')]
        db_user = os.environ[db_user_env if db_user_env is not None else (env_prefix + 'db_user')]
        db_host = os.environ[db_host_env if db_host_env is not None else (env_prefix + 'db_host')]
        db_credentials = os.environ[
            db_credentials_env if db_credentials_env is not None else (env_prefix + 'db_credentials')]
        conn_string = f'postgresql://{db_user}:{db_credentials}@{db_host}:5432/{db_name}'
    elif db_type == 'bigquery':
        conn_string = f'bigquery://{os.environ["gcp_project"]}'
    else:
        raise ValueError('db_type must be "postgres" or "bigquery"')

    return create_engine(conn_string)


def get_sqa_table(engine, schema, table):
    """Retrieve a table under the given schema

    Args:
        engine: active database connection
        schema: schema for the table
        table: desired table

    Returns: the desired table

    """
    if str(engine.engine.url).split('://')[0] == 'bigquery':
        return Table(f'{schema}.{table}', MetaData(bind=engine), autoload=True)
    else:
        metadata = MetaData(schema=schema)
        metadata.reflect(engine, only=['specifications'])
        return metadata.tables[f'{schema}.{table}']


class IndentLogger(logging.LoggerAdapter):
    """Logger class which includes indents to indicate the height of the frame stack. Makes it easy to distinguish activity of inner and outer functions.

    use this adapter with:
        import logging
        logger = IndentLogger(logging.getLogger(''), {})
        logger.info(...)
    """
    @staticmethod
    def indent():
        """Determine the indent level to use based on the height of the frame stack

        Returns: integer representing how much to indent the logging message

        """
        indentation_level = len(traceback.extract_stack())
        return indentation_level - 5

    def process(self, msg, kwargs):
        """Format the msg using the desired indent level

        Args:
            msg: logging message
            kwargs: keyword arguments to be used in logging

        Returns:

        """
        return "{}{}".format('-' * self.indent() + "|", msg), kwargs


def setup(name, log_level="info"):
    """Set up logging and determine the output filepath based on the passed name parameter

    The logging levels are ['debug', 'info', 'warning', 'error', 'critical'] and are nested, so all levels after the chosen level will also be printed to the screen
    For example, selecting 'warning' will result in 'warning', 'error', and 'critical' messages being printed to the screen.

    Args:
        name: name of the subdirectory within the output directory which should contain output
        log_level: What level of log message to output to the screen. Log files always contain up to debug level

    Returns: output directory where output can be saved

    """
    outdir = os.path.join("covid_model", "output", name)
    os.makedirs(outdir, exist_ok=True)

    # options for log level
    log_dict = {'debug': logging.DEBUG,
                'info': logging.INFO,
                'warning': logging.WARNING,
                'error': logging.ERROR,
                'critical': logging.CRITICAL}

    # set up logging to file (file always does LEVEL level )
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


def get_filepath_prefix(outdir=None, tags=None):
    """construct a prefix for a filepath that includes the current datetime, and also possibly some model tags

    Args:
        outdir: output directory where this file will go. gets prepended to filepath prefix if present
        tags: model tags to include in the file path. get appended to the filepath prefix if present

    Returns: a filepath prefix.

    """
    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = join(outdir, now) if outdir is not None else now
    if tags:
        prefix += f'_{"_".join(str(key) + "_" + str(val) for key, val in tags.items())}'
    return prefix + '_'
