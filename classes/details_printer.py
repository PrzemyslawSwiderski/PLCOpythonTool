import logging

from utils.helpers import get_top_abs_correlations


class ConsoleDataSetDetailsPrinter:
    def __init__(self, data_set):
        self.logger = logging
        self.data_set = data_set

    def print_data_set_correlation(self):
        self.logger.info("\n_________________________"
                         "\nDataSet Correlation:"
                         f"\n{self.data_set.corr()}")

    def print_data_set_stats(self):
        self.logger.info("\n_________________________"
                         "\nDataSet:"
                         f"\n{self.data_set}"
                         "\nDataSet Shape:"
                         f"\n{self.data_set.shape}"
                         "\nDataSet Stats:"
                         f"\n{self.data_set.describe(include='all')}")

    def print_top_correlations(self, n=20):
        self.logger.info("\n_________________________"
                         "\nTop Absolute Correlations"
                         f"\n{get_top_abs_correlations(self.data_set, n)}")

    def print_all(self):
        self.print_data_set_stats()
        self.print_data_set_correlation()
        self.print_top_correlations()
