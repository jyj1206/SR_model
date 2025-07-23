import logging
import os
import io
from datetime import datetime
from contextlib import redirect_stdout
from torchsummary import summary


def setup_logger(log_dir="logs", log_name=None):
    os.makedirs(log_dir, exist_ok=True)
    
    if log_name is None:
        log_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_model_summary(model, input_size, logger):
    buf = io.StringIO()
    with redirect_stdout(buf):  # summary의 print 출력을 캡처
        summary(model, input_size=input_size)
    summary_str = buf.getvalue()
    logger.info("\n" + summary_str)
