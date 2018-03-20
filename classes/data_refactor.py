import logging

import pandas

from classes.details_printer import ConsoleDataSetDetailsPrinter


def refactor_data_set_to_numeric(data_set):
    data_set = data_set.apply(pandas.to_numeric, errors="coerce")
    data_set.dropna(axis=1, how="all", inplace=True)
    logging.info("\nData_set as numeric stats: ")
    ConsoleDataSetDetailsPrinter(data_set).print_all()
    return data_set


def prepare_data_set_to_predict_mortality(data_set):
    data_set['dth_days'].fillna(5000, inplace=True)
    logging.info("\nPrepared data_set to predict mortality: ")
    ConsoleDataSetDetailsPrinter(data_set).print_all()
    return data_set
