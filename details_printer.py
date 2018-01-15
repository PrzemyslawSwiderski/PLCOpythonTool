import logging

from helpers import get_top_abs_correlations


class DefaultDetailsPrinter:
    def __init__(self, fetcher_object):
        self.fetcher_object = fetcher_object
        self.logger = logging

    def print_data_set_correlation(self):
        self.logger.info("_________________________")
        self.logger.info("DataSet Correlation:")
        self.logger.info(self.fetcher_object.data_set.corr())

    def print_normalized_data_set_correlation(self):
        self.logger.info("_________________________")
        self.logger.info("Normalized DataSet Correlation:")
        self.logger.info(self.fetcher_object.data_set_normalized.corr())

    def print_data_set_stats(self):
        self.logger.info("_________________________")
        self.logger.info("DataSet:")
        self.logger.info(self.fetcher_object.data_set)
        self.logger.info("DataSet Shape:")
        self.logger.info(self.fetcher_object.data_set.shape)
        self.logger.info("DataSet Stats:")
        self.logger.info(self.fetcher_object.data_set.describe(include='all'))

    def print_normalized_data_set_stats(self):
        self.logger.info("_________________________")
        self.logger.info("Sample Normalized Train DataSet:")
        self.logger.info(self.fetcher_object.data_set_normalized)
        self.logger.info("Normalized Train DataSet Shape:")
        self.logger.info(self.fetcher_object.data_set_normalized.shape)
        self.logger.info("Normalized Train DataSet Stats:")
        self.logger.info(self.fetcher_object.data_set_normalized.describe())

    def print_all(self):
        self.logger.info("PRENORMALIZED DATASET")
        self.logger.info("_________________________")
        self.print_all_data_set()
        self.logger.info("NORMALIZED DATASET")
        self.logger.info("_________________________")
        self.print_all_data_set_normalized()

    def print_all_data_set(self):
        self.print_data_set_correlation()
        self.print_top_correlations(self.fetcher_object.data_set)
        self.print_data_set_stats()

    def print_all_data_set_normalized(self):
        self.print_normalized_data_set_correlation()
        self.print_normalized_data_set_stats()

    def print_top_correlations(self, data_set, n=20):
        self.logger.info("_________________________")
        self.logger.info("Top Absolute Correlations")
        self.logger.info(get_top_abs_correlations(data_set, n))
