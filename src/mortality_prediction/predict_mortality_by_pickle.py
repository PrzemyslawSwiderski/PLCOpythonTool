import argparse
import os

from sklearn.externals import joblib

from classes.mlp_predictor import MLPPredictor
from common_config.common_config import LOGS_CATALOG_PATH, PICKLES_PATH
from common_config.logger_conf import configure_logging
from mortality_prediction.mortality_prediction_net_config import config


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    parser = argparse.ArgumentParser(description="Script creates neural net model and predicts mortality")
    parser.add_argument("p", help="Pickled data_set file to be load")
    pickled_data_file = parser.parse_args().p

    data_processor = joblib.load(os.path.join(PICKLES_PATH, pickled_data_file))
    predictor = MLPPredictor(config, data_processor)
    predictor.train_net()
    predictor.predict_by_test_data_from_config()


if __name__ == "__main__":
    main()
