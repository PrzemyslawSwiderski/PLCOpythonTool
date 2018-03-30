import os

from sklearn.externals import joblib

from common_config.common_config import PICKLES_PATH
from treatment_prediction.treatment_data_processor import TreatmentDataProcessor
from treatment_prediction.treatment_prediction_net_config import config
from utils.logging_helper import log_train_results_of_MLP


class TreatmentPredictionWithMLPC:
    def __init__(self, data_processor=TreatmentDataProcessor()):
        self.data_processor = data_processor
        self.mlp = config["neural_net_def"]

    def save_net(self):
        joblib.dump(self.mlp, os.path.join(PICKLES_PATH, config["saved_net_name"]))

    def load_net(self):
        self.mlp = joblib.load(os.path.join(PICKLES_PATH, config["saved_net_name"]))

    def train_network(self):
        self.data_processor.process_data()
        self.mlp.fit(self.data_processor.X_train_, self.data_processor.Y_train_)
        if self.mlp.solver != "lbfgs":
            self.mlp.partial_fit(self.data_processor.X_test_, self.data_processor.Y_test_)
        log_train_results_of_MLP(self.mlp, self.data_processor.X_test_, self.data_processor.Y_test_)

    def train_net_and_save(self):
        self.train_network()
        self.save_net()
