import os

from sklearn.externals import joblib

from common_config.common_config import PICKLES_PATH


class DataSetPickler:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def process_and_pickle_data_processor(self):
        self.data_processor.process_data()
        joblib.dump(self.data_processor, os.path.join(PICKLES_PATH, self.data_processor.config["pickle_file_name"]))
