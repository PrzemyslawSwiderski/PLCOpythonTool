import argparse
import os

from sklearn.externals import joblib

from common_config.config import LOGS_CATALOG_PATH, PICKLES_PATH, FEATURES_STATS_PATH
from common_config.logger_conf import configure_logging
from utils.helpers import get_top_abs_correlations


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    parser = argparse.ArgumentParser(description="Script finds data_set features correlations by loaded pickle and "
                                                 "saves it to features_stats catalog")
    parser.add_argument("p", help="Pickled data_set file to be load instead of query DB")
    pickled_data_file = parser.parse_args().p

    data_set = joblib.load(os.path.join(PICKLES_PATH, f"{pickled_data_file}.pkl"))

    correlations = get_top_abs_correlations(data_set, -1)
    with open(os.path.join(FEATURES_STATS_PATH, f"correlations_{pickled_data_file}.txt"), 'w') as file:
        file.write(correlations.to_string())


if __name__ == "__main__":
    main()
