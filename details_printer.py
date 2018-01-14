import logging

from helpers import get_top_abs_correlations


class DefaultDetailsPrinter:
    def __init__(self, fetcher_object):
        self.fetcher_object = fetcher_object
        self.logger = logging

    def print_data_set_correlation(self):
        self.logger.info("DataSet Correlation:")
        self.logger.info(self.fetcher_object.data_set.corr())

    def print_data_set_stats(self):
        self.logger.info("DataSet:")
        self.logger.info(self.fetcher_object.data_set)
        self.logger.info("DataSet Shape:")
        self.logger.info(self.fetcher_object.data_set.shape)
        self.logger.info("DataSet Stats:")
        self.logger.info(self.fetcher_object.data_set.describe(include='all'))

    def print_scaled_data_set_stats(self):
        self.logger.info("Sample Normalized Train DataSet:")
        self.logger.info(self.fetcher_object.data_set_normalized)
        self.logger.info("Normalized Train DataSet Shape:")
        self.logger.info(self.fetcher_object.data_set_normalized.shape)
        self.logger.info("Normalized Train DataSet Stats:")
        self.logger.info(self.fetcher_object.data_set_normalized.describe())

    def print_all(self):
        self.print_data_set_correlation()
        self.print_top_correlations()
        self.print_data_set_stats()
        self.print_scaled_data_set_stats()

    def print_top_correlations(self, n=5):
        print("Top Absolute Correlations")
        print(get_top_abs_correlations(self.fetcher_object.data_set, n))
