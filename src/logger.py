import logging
import os
from datetime import datetime

log_file = os.path.join(os.getcwd(), "log", f"{datetime.now().strftime('%m_%d_%Y_%H_%M')}.log")
# logs_path=os.path.join
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(lineno)d - %(message)s',
    filename=log_file
)


if __name__ == "__main__":
    logging.info("Logging started")
