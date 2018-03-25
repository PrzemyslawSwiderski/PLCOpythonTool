import logging
from abc import abstractmethod

from pandas import DataFrame
from sklearn import model_selection

from classes.details_printer import ConsoleDataSetDetailsPrinter
from utils.helpers import refactor_data_set_to_numeric, pretty_print_dict, exclude_boundary_values


class CommonDataProcessor:
    def __init__(self, config, input_data_set=DataFrame()):
        self.config = config
        self.data_set = input_data_set
        if 'scaler' in config:
            self.scaler = config['scaler']

    def process_data(self):
        self.data_set = refactor_data_set_to_numeric(self.data_set)
        self.optimize_values()
        self.filter_data_set()
        self.split_data_set()
        if hasattr(self, "scaler"):
            self.scale_data_set()
        self.print_info()

    @abstractmethod
    def optimize_values(self):
        pass

    def filter_data_set(self):
        for value in self.config["boundary_values_to_exclude"]:
            self.data_set = exclude_boundary_values(self.data_set, value["feature_name"],
                                                    value["boundary_scale_value"])

        self.data_set = self.data_set[self.config["features_to_preserve_after_preprocessing"]]
        self.filtered_data_set_ = self.data_set

    def scale_data_with_Y(self):
        self.train_dataset_scaled_ = self.scaler.fit_transform(self.train_dataset_)
        self.test_dataset_scaled_ = self.scaler.transform(self.test_dataset_)
        self.X_train_ = self.train_dataset_scaled_[:, :-1]
        self.X_test_ = self.test_dataset_scaled_[:, :-1]
        self.Y_train_ = self.train_dataset_scaled_[:, -1]
        self.Y_test_ = self.test_dataset_scaled_[:, -1]

    def scale_data_without_Y(self):
        self.train_dataset_scaled_ = self.scaler.fit_transform(self.train_dataset_.values[:, :-1])
        self.test_dataset_scaled_ = self.scaler.transform(self.test_dataset_.values[:, :-1])
        self.X_train_ = self.train_dataset_scaled_
        self.X_test_ = self.test_dataset_scaled_
        self.Y_train_ = self.train_dataset_.values[:, -1]
        self.Y_test_ = self.test_dataset_.values[:, -1]

    def scale_data_set(self):
        if self.config["should_scale_Y"]:
            self.scale_data_with_Y()
        else:
            self.scale_data_without_Y()

    def print_info(self):
        logging.info(f"\nPrepared data_set to predict {self.config['prediction_name']}: ")
        details_printer = ConsoleDataSetDetailsPrinter(self.data_set)
        details_printer.print_all()
        details_printer.print_correlations_of_features(self.config["features_to_print_correlations"])
        logging.info("\nLoaded processor's properties: "
                     f"\n{pretty_print_dict(self.config)}")

    def split_data_set(self):
        X = self.data_set.columns[:-1]
        Y = self.data_set.columns[-1]
        split_value = self.config["random_state_split_value"]
        train_test_split = model_selection.train_test_split(self.data_set[X], self.data_set[Y],
                                                            random_state=split_value,
                                                            test_size=self.config["validation_size"])
        self.X_train_, self.X_test_, self.Y_train_, self.Y_test_ = train_test_split
        self.train_dataset_ = self.X_train_.assign(e=self.Y_train_.values)
        self.train_dataset_.rename(columns={"e": self.Y_train_.name}, inplace=True)
        self.test_dataset_ = self.X_test_.assign(e=self.Y_test_.values)
        self.test_dataset_.rename(columns={"e": self.Y_test_.name}, inplace=True)
