import argparse
import os

from sklearn.externals import joblib

from classes.data_refactor import refactor_data_set_to_numeric, prepare_data_set_to_predict_mortality
from classes.mysql_fetcher import MySqlFetcher
from config.config import QUERIES_PATH, LOGS_CATALOG_PATH, PICKLES_PATH
from config.logger_conf import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Script executes SQL query file from queries catalog and saves "
                                                 "pickled data in pickles catalog")
    parser.add_argument('queryFile', help='SQL Query File name in queries catalog to be executed')
    query_file = parser.parse_args().queryFile
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    ms = MySqlFetcher()
    data_set = ms.run_select_query_from_file(
        os.path.join(QUERIES_PATH, f"{query_file}.sql"))
    data_set = refactor_data_set_to_numeric(data_set)
    data_set = prepare_data_set_to_predict_mortality(data_set)

    joblib.dump(data_set, f'{PICKLES_PATH}/saved_data_set_{query_file}.pkl')


if __name__ == "__main__":
    main()
