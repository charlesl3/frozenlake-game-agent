import logging
import os

def get_logger(name: str, log_file: str = "logs/train.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
