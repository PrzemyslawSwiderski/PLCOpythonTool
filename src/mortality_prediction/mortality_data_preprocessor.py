import logging

import numpy
import pandas

from data_preprocessor import DefaultDataPreprocessor
from details_printer import ConsoleDataSetDetailsPrinter
from helpers import exclude_boundary_values, refactor_data_set_to_numeric, pretty_print_dict
from mortality_script_config import config


class MortalityDataPreprocessor(DefaultDataPreprocessor):

    def preprocess_data(self, data_set):
        data_set = refactor_data_set_to_numeric(data_set)
        return self.prepare_data_set_to_predict_mortality(data_set)

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

    def prepare_data_set_to_predict_mortality(self, data_set):
        data_set = self.fill_missing_dth_days(data_set)
        data_set = exclude_boundary_values(data_set, "dth_days", config["dth_days_boundary_scale_value"])
        data_set = exclude_boundary_values(data_set, "dx_psa", config["dx_psa_boundary_scale_value"])
        # scaler = StandardScaler()
        # data_set_scaled = pandas.DataFrame(scaler.fit_transform(data_set))
        logging.info("\nPrepared data_set to predict mortality: ")
        details_printer = ConsoleDataSetDetailsPrinter(data_set)
        details_printer.print_all()
        details_printer.print_correlations_of_feature("dth_days")
        logging.info("\nLoaded preprocessor's properties: "
                     f"\n{pretty_print_dict(config)}")
        return data_set
