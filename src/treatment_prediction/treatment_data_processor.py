import os

from classes.data_processor import CommonDataProcessor
from classes.mysql_fetcher import MySqlFetcher
from common_config.common_config import QUERIES_PATH
from treatment_prediction.treatment_data_processor_config import config


class TreatmentDataProcessor(CommonDataProcessor):
    def __init__(self):
        CommonDataProcessor.__init__(self, config)
        self.get_data_from_database()

    def optimize_values(self):
        self.data_set.fillna(0, inplace=True)

    def get_data_from_database(self):
        ms = MySqlFetcher()
        self.data_set = ms.run_select_query_from_file(os.path.join(QUERIES_PATH, config["query_file_name"]))
        ms.close_connection()
