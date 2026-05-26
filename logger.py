"""
logger.py

Custom logging utility for the Agentic AutoML pipeline.
Provides standardized console output for pipeline stages and metrics.
Usage:
    >>> from logger import log_stage, log_metric
    >>> log_stage("Initialization")
"""
import logging
import warnings
import sys
import traceback

def setup_logger():
    # Suppress all verbose library warnings to save tokens and reduce noise
    warnings.filterwarnings('ignore')
    
    # Configure custom logger
    logger = logging.getLogger('AgenticAutoML')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Add console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

def log_info(msg: str):
    logger.info(msg)

def enable_file_logging(log_file_path: str = "automl.log"):
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def log_stage(stage_name: str):
    logger.info(f"--- STAGE: {stage_name} ---")

def log_metric(metric_name: str, value: float):
    logger.info(f"METRIC - {metric_name}: {value:.4f}")

def log_error(error_msg: str, exc_info: Exception = None):
    logger.error(f"FAILED: {error_msg}")
    if exc_info:
        tb = "".join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
        logger.error(f"TRACEBACK:\n{tb}")

# Utility to silence verbose stdout from third-party C libraries or print statements
import os
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield
