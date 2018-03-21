import argparse
import os

from sklearn.externals import joblib

from classes.details_printer import ConsoleDataSetDetailsPrinter
from common_config.config import LOGS_CATALOG_PATH, PICKLES_PATH
from common_config.logger_conf import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Script prints data_set stats by loaded pickle")
    parser.add_argument("p", help="Pickled data_set file to be load instead of query DB")
    pickled_data_file = parser.parse_args().p
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))

    data_set = joblib.load(os.path.join(PICKLES_PATH, f"{pickled_data_file}.pkl"))
    dp = ConsoleDataSetDetailsPrinter(data_set)
    dp.print_all()


if __name__ == "__main__":
    main()
