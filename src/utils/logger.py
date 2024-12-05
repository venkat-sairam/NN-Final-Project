from datetime import datetime
import logging
import os

timestamp_format = "%d-%m-%y_%H-%M-%S"

CURR_TIMESTAMP = f"{datetime.now().strftime(timestamp_format)}"
LOGS_DIRECTORY_NAME = "Log_Files"
CWD = os.getcwd()

directory_path = os.path.join(CWD, LOGS_DIRECTORY_NAME)
os.makedirs(directory_path, exist_ok=True)

LOG_FILE_NAME = f"{CURR_TIMESTAMP}.log"
LOG_FILE_PATH = os.path.join(directory_path, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(filename)s %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Add StreamHandler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set level for StreamHandler (optional)

# Define the log format for the console output
console_formatter = logging.Formatter("[%(asctime)s] %(filename)s %(lineno)d  - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Add StreamHandler to the root logger
logging.getLogger().addHandler(console_handler)