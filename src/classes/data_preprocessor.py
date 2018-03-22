import logging
from abc import abstractmethod

from classes.details_printer import ConsoleDataSetDetailsPrinter
from utils.helpers import refactor_data_set_to_numeric, pretty_print_dict, exclude_boundary_values


class CommonDataPreprocessor:
    def __init__(self, config):
        self.config = config
        if 'scaler' in config:
            self.scaler = config['scaler']

    def preprocess_data(self, data_set):
        data_set = refactor_data_set_to_numeric(data_set)
        data_set = self.optimize_values(data_set)
        data_set = self.filter_data_set(data_set)
        self.print_info(data_set)
        if hasattr(self, "scaler"):
            data_set = self.scale_data_set(data_set)

        return data_set

    @abstractmethod
    def optimize_values(self, data_set):
        pass

    def filter_data_set(self, data_set):
        for value in self.config["boundary_values_to_exclude"]:
            data_set = exclude_boundary_values(data_set, value["feature_name"], value["boundary_scale_value"])

        data_set = data_set[self.config["features_to_preserve_after_preprocessing"]]
        return data_set

    def scale_data_set(self, data_set):
        features_to_be_scaled = data_set.columns[:-1]
        data_set[features_to_be_scaled] = self.scaler.fit_transform(data_set[features_to_be_scaled])
        return data_set

    def print_info(self, data_set):
        logging.info(f"\nPrepared data_set to predict {self.config['prediction_name']}: ")
        details_printer = ConsoleDataSetDetailsPrinter(data_set)
        details_printer.print_all()
        details_printer.print_correlations_of_features(self.config["features_to_print_correlations"])
        logging.info("\nLoaded preprocessor's properties: "
                     f"\n{pretty_print_dict(self.config)}")
