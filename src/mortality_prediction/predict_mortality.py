import os

from common_config.common_config import LOGS_CATALOG_PATH
from common_config.logger_conf import configure_logging
from mortality_prediction.mortality_prediction_net_with_mlpr import MortalityPredictionWithMLPR


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    predictor = MortalityPredictionWithMLPR()
    predictor.train_network()
    predictor.predict_by_test_data_from_config()


if __name__ == "__main__":
    main()
