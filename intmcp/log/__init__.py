"""Logging module """
import logging


INFO1 = logging.INFO - 1
INFO2 = logging.INFO - 2
DEBUG1 = logging.DEBUG - 1
DEBUG2 = logging.DEBUG - 2


def config_logger(level: int, **kwargs):
    """Set the logging level """
    logging.basicConfig(level=level, format='%(message)s', **kwargs)
    logging.addLevelName(INFO1, "INFO1")
    logging.addLevelName(INFO2, "INFO2")
    logging.addLevelName(DEBUG1, "DEBUG1")
    logging.addLevelName(DEBUG2, "DEBUG2")
    logging.info("Configuring logger")
