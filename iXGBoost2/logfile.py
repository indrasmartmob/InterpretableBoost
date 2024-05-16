import glob
import sys
from pathlib import Path
file_path = Path(__file__)
parent_directory_path = file_path.parent.parent
logurupath=str(parent_directory_path)+"\loguru-master"
sys.path.insert(0,logurupath)
from loguru import logger

enable_disable_str="iXGBoost2"

def log_enable():
    logger.enable(enable_disable_str)
    return None

def log_disable():
    logger.disable(enable_disable_str)
    return None

# logger.add(writer, format="{message}")
# logger.enable(None)
# logger.debug("yes")
# logger.disable(None)
# logger.debug("nope")
stop_the_logger_with_every_files=True
if stop_the_logger_with_every_files==True:
    logger.disable(enable_disable_str)