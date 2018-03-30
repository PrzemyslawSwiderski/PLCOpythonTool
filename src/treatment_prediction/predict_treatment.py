import os

from common_config.common_config import LOGS_CATALOG_PATH
from common_config.logger_conf import configure_logging
from treatment_prediction.treatment_prediction_net_with_mlpc import TreatmentPredictionWithMLPC
from mortality_prediction.mortality_prediction_net_with_mlpr import MortalityPredictionWithMLPR


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    predictor = TreatmentPredictionWithMLPC()
    predictor.train_network()
    pass


if __name__ == "__main__":
    main()
