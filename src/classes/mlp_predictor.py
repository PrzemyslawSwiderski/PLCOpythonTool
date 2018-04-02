import os

from sklearn.externals import joblib

from common_config.common_config import PICKLES_PATH
from utils.helpers import exclude_data_frame_columns
from utils.logging_helper import log_train_results_of_MLP, log_predict_result


class MLPPredictor:
    def __init__(self, config, data_processor):
        self.data_processor = data_processor
        self.config = config
        self.mlp = self.config["neural_net_def"]

    def save_net(self):
        joblib.dump(self.mlp, os.path.join(PICKLES_PATH, self.config["saved_net_name"]))

    def load_net(self):
        self.mlp = joblib.load(os.path.join(PICKLES_PATH, self.config["saved_net_name"]))

    def train_network(self):
        self.data_processor.process_data()
        self.mlp.fit(self.data_processor.X_train_, self.data_processor.Y_train_)
        if self.mlp.solver != "lbfgs":
            self.mlp.partial_fit(self.data_processor.X_test_, self.data_processor.Y_test_)
        log_train_results_of_MLP(self.mlp, self.data_processor.X_test_, self.data_processor.Y_test_)

    def train_net_and_save(self):
        self.train_network()
        self.save_net()

    def predict_by_input(self, input_tab):
        input_scaled = self.data_processor.transform_input_tab(input_tab[self.data_processor.data_set.columns])
        predicted_value = self.mlp.predict(
            exclude_data_frame_columns(input_scaled, self.data_processor.config["features_to_predict"]))
        input_scaled[self.data_processor.config["features_to_predict"]] = predicted_value
        output_tab = self.data_processor.inverse_transform_input_tab(input_scaled)
        log_predict_result(input_tab[self.data_processor.config["features_to_predict"]].values,
                           output_tab[self.data_processor.config["features_to_predict"]].values)

    def predict_by_test_data_from_config(self):
        self.predict_by_input(self.config["test_input_data"])
