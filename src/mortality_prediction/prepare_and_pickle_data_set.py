import os

from classes.data_set_pickler import DataSetPickler
from common_config.common_config import LOGS_CATALOG_PATH
from common_config.logger_conf import configure_logging
from mortality_prediction.mortality_data_processor import MortalityDataProcessor


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    data_pickler = DataSetPickler(MortalityDataProcessor())
    data_pickler.process_and_pickle_data_processor()


if __name__ == "__main__":
    main()
