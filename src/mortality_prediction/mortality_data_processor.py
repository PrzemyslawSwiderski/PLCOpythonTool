import os

import numpy
import pandas

from classes.data_processor import CommonDataProcessor
from classes.mysql_fetcher import MySqlFetcher
from common_config.common_config import QUERIES_PATH
from mortality_prediction.mortality_data_processor_config import config


class MortalityDataProcessor(CommonDataProcessor):
    def __init__(self):
        CommonDataProcessor.__init__(self, config)
        self.get_data_from_database()

    def fill_missing_dth_days(self):
        years_left = numpy.zeros(len(self.data_set['age']))
        for index, value in enumerate(self.data_set['age']):
            if self.data_set['pros_exitage'].values[index] >= config["average_life_duration"]:
                years_left[index] = self.data_set['pros_exitage'].values[index] - value
            else:
                years_left[index] = config["average_life_duration"] - value

        days_left = years_left * config["days_in_year"]
        self.data_set['dth_days'].fillna(pandas.Series(days_left), inplace=True)

    def optimize_values(self):
        self.fill_missing_dth_days()
        if 'surg_age' in self.data_set:
            self.data_set['surg_age'].fillna(self.data_set['surg_age'].mean(), inplace=True)
        self.data_set.fillna(0, inplace=True)

    def get_data_from_database(self):
        ms = MySqlFetcher()
        self.data_set = ms.run_select_query_from_file(os.path.join(QUERIES_PATH, config["query_file_name"]))
        ms.close_connection()
