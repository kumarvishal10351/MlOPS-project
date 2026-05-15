import logging 
import os
from datetime import datetime

LOG_FILE = os.path.join(os.getcwd(), 'logs', f'{datetime.now().strftime("%Y-%m-%d")}.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
LOG_FILE_PATH = os.path.join(os.getcwd(), 'logs')

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    logging.info("This is an info message.")
    logging.error("This is an error message.")