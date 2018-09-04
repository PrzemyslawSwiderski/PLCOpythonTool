import os

from classes.mlp_predictor import MLPPredictor
from common_config.common_config import LOGS_CATALOG_PATH
from common_config.logger_conf import configure_logging
from mortality_prediction.mortality_data_processor import MortalityDataProcessor
from mortality_prediction.mortality_prediction_net_config import config


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    mortality_predictor = MLPPredictor(config, MortalityDataProcessor())
    mortality_predictor.train_net()
    mortality_predictor.predict_by_test_data_from_config()


if __name__ == "__main__":
    main()
