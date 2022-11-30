### Python Standard Library ###
from os.path import join
import datetime as dt
import logging
import traceback
import os
### Third Party Imports ###
import matplotlib
### Local Imports ###


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


def get_filepath_prefix(outdir = None):
    if outdir:
        return join(outdir, dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_')
    else:
        return dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_'