import logging
import os

import numpy
from pandas import DataFrame
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

from common_config.common_config import PICKLES_PATH
from mortality_prediction.mortality_data_processor import MortalityDataProcessor
from mortality_prediction.mortality_prediction_net_config import config


class MortalityPredictionWithMLPR:
    def __init__(self, data_processor=MortalityDataProcessor()):
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
        self.log_train_results_MLPRegressor(self.data_processor.X_test_, self.data_processor.Y_test_)

    def train_net_and_save(self):
        self.train_network()
        self.save_net()

    def predict_by_input(self, input_tab):
        input_scaled = self.data_processor.transform_input_tab(input_tab)
        predicted_value = self.mlp.predict([input_scaled[:-1]])
        input_scaled[-1] = predicted_value[0]
        output_tab = self.data_processor.inverse_transform_input_tab(input_scaled)
        self.log_predict_result(input_tab, output_tab[-1])

    def predict_by_test_data_from_config(self):
        self.predict_by_input(config["test_input_data"])

    def log_predict_result(self, input_tab, predicted):
        target = input_tab[-1]
        abs_diff = abs(target - predicted)
        logging.info(f"Input parameters: {input_tab[:-1]}")
        logging.info(f"Predicted value of dth_days: {predicted}")
        logging.info(f"Target value of dth_days: {target}")
        logging.info(f"Predicted value and target abs diff: {abs_diff}")

    def log_train_results_MLPRegressor(self, X_validation, Y_validation):
        logging.info("Real / Predicted values:")
        mlp_predicted_values = self.mlp.predict(X_validation)
        logging.info(DataFrame({'predicted': mlp_predicted_values, 'real': Y_validation}))
        logging.info(f"Estimator:\n{self.mlp}")
        logging.info(f"Number of iterations: {self.mlp.n_iter_}")
        logging.info("Score:")
        logging.info(self.mlp.score(X_validation, Y_validation))
        logging.info("Mean squared error:")
        logging.info(mean_squared_error(mlp_predicted_values, Y_validation))
