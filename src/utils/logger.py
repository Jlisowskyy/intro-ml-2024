import logging, sys


def get_fastapi_logger() -> logging.Logger:
    """
    Get a FastAPI compatible logger object

    :return: Logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger
