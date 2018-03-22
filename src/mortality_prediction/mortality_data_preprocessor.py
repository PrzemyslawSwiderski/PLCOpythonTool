import numpy
import pandas

from classes.data_preprocessor import CommonDataPreprocessor
from mortality_prediction.mortality_script_config import config


class MortalityDataPreprocessor(CommonDataPreprocessor):
    def __init__(self):
        super(MortalityDataPreprocessor, self).__init__(config)

    def fill_missing_dth_days(self, data_set):
        years_left = numpy.zeros(len(data_set['age']))
        for index, value in enumerate(data_set['age']):
            if data_set['pros_exitage'].values[index] >= config["average_life_duration"]:
                years_left[index] = data_set['pros_exitage'].values[index] - value
            else:
                years_left[index] = config["average_life_duration"] - value

        days_left = years_left * config["days_in_year"]
        data_set['dth_days'].fillna(pandas.Series(days_left), inplace=True)
        return data_set

    def optimize_values(self, data_set):
        data_set = self.fill_missing_dth_days(data_set)
        data_set['surg_age'].fillna(data_set['surg_age'].mean(), inplace=True)
        data_set.fillna(0, inplace=True)

        return data_set
