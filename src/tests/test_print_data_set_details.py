import os
import unittest

from classes.details_printer import ConsoleDataSetDetailsPrinter
from classes.mysql_fetcher import MySqlFetcher
from common_config.config import QUERIES_PATH, LOGS_CATALOG_PATH
from common_config.logger_conf import configure_logging


class TestPrintDataSetDetails(unittest.TestCase):
    def setUp(self):
        configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
        ms = MySqlFetcher()
        data_set = ms.run_select_query_from_file(
            os.path.join(QUERIES_PATH, "select_to_predict_mortality_stats_query.sql"))
        self.dp = ConsoleDataSetDetailsPrinter(data_set)

    def test_print_all(self):
        self.dp.print_all()

    def test_print_data_set_correlation(self):
        self.dp.print_data_set_correlation()

    def test_print_data_set_stats(self):
        self.dp.print_data_set_stats()

    def test_print_top_correlations(self):
        self.dp.print_top_correlations()

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
