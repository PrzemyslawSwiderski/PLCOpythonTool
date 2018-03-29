import logging
from abc import abstractmethod

import numpy
from pandas import DataFrame
from sklearn import model_selection

from classes.details_printer import ConsoleDataSetDetailsPrinter
from utils.helpers import refactor_data_set_to_numeric, pretty_print_dict, exclude_boundary_values


class CommonDataProcessor:
    def __init__(self, config, input_data_set=DataFrame()):
        self.config = config
        self.data_set = input_data_set

    def process_data(self):
        self.data_set = refactor_data_set_to_numeric(self.data_set)
        self.optimize_values()
        self.filter_data_set()
        self.split_data_set()
        if 'PCA_transform' in self.config:
            self.invoke_PCA_transform()
        if 'scaler' in self.config:
            self.scale_data_set()

        logging.info(f"\nData set after processing description: "
                     "\nX_TRAIN:"
                     f"\n{DataFrame(self.X_train_).describe()}"
                     "\nY_TRAIN:"
                     f"\n{DataFrame(self.Y_train_).describe()}")
        self.print_info()

    @abstractmethod
    def optimize_values(self):
        pass

    def filter_data_set(self):
        for value in self.config["boundary_values_to_exclude"]:
            self.data_set = exclude_boundary_values(self.data_set, value["feature_name"],
                                                    value["boundary_scale_value"])

        self.data_set.drop(self.config["features_to_exclude_after_preprocessing"], axis=1, inplace=True)
        self.filtered_data_set_ = self.data_set

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

    def invoke_PCA_transform(self):
        self.pca_ = self.config["PCA_transform"]["PCA_object"]
        self.X_train_ = self.pca_.fit_transform(self.X_train_)
        self.X_test_ = self.pca_.transform(self.X_test_)
        logging.info(f"\nX train after PCA transformation: "
                     f"{self.X_train_}")

    def scale_data_set(self):
        self.scaler_ = self.config["scaler"]
        self.scaler_.scale_data_set_in_data_processor(self)

    def transform_input_tab(self, input_tab):
        if 'PCA_transform' in self.config:
            pca_tab = self.pca_.transform([input_tab[:-1]])[0].tolist()
            pca_tab.append(input_tab[-1])
            input_tab = pca_tab
        if 'scaler' in self.config:
            input_tab = self.scaler_.scaler_transform(input_tab)
        return input_tab

    def inverse_transform_input_tab(self, input_tab):
        if 'scaler' in self.config:
            input_tab = self.scaler_.scaler_inverse_transform(input_tab)
        return input_tab
