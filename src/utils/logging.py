import logging 
import os 
import time 

def experiment_args_logger(logger_save_name, root_dir, level=logging.INFO, screen=True, tofile=True):
    """Function for creating a logfile for the experimental config arguments & app config arugments."""
    lg = logging.getLogger(logger_save_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)

    log_time = get_timestamp()
    if tofile:
        log_file = os.path.join(root_dir, "{}.log".format(logger_save_name))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg, log_time


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime

