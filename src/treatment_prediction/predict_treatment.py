import os

from classes.mlp_predictor import MLPPredictor
from common_config.common_config import LOGS_CATALOG_PATH
from common_config.logger_conf import configure_logging
from treatment_prediction.treatment_data_processor import TreatmentDataProcessor
from treatment_prediction.treatment_prediction_net_config import config


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    treatment_predictor = MLPPredictor(config, TreatmentDataProcessor())
    treatment_predictor.train_network()


if __name__ == "__main__":
    main()
