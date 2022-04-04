import logging
import os.path
import sys

from parameters import model_dst

file_url = os.path.join(model_dst, 'debug.log')

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(file_url),
        logging.StreamHandler(sys.stdout)
    ]
)


if __name__ == '__main__':
    logging.debug('debug message')
    logging.info('info message')
    logging.warning('warn message')
    logging.error('error message')
    logging.critical('critical message')