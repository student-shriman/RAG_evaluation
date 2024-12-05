# logger.py
import logging

def setup_logger(module_name, log_file):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Create a file handler for this module's log file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    return logger
        