import os

from sklearn.externals import joblib

from classes.data_preprocessor import CommonDataPreprocessor
from classes.mysql_fetcher import MySqlFetcher
from common_config import logger_conf
from common_config.common_config import LOGS_CATALOG_PATH, QUERIES_PATH, PICKLES_PATH


class DataSetPickler:

    def __init__(self, query_file, data_preprocessor=CommonDataPreprocessor):
        logger_conf.configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
        self.data_preprocessor = data_preprocessor
        self.query_file = query_file

    def get_data_from_database(self):
        ms = MySqlFetcher()
        data_set = ms.run_select_query_from_file(
            os.path.join(QUERIES_PATH, f"{self.query_file}.sql"))
        ms.close_connection()
        return data_set

    def preprocess_and_pickle_data_set(self):
        data_set = self.get_data_from_database()
        data_set = self.data_preprocessor.preprocess_data(data_set)
        joblib.dump(data_set, f"{PICKLES_PATH}/saved_data_set_{self.query_file}.pkl")
