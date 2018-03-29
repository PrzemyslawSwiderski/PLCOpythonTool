import numpy
from pandas import DataFrame


class CommonDataProcessorScaler:

    def __init__(self, config):
        self.config = config
        self.scaler_X = self.config["scaler_X"]
        self.scaler_Y = self.config["scaler_Y"]

    def scale_data_with_Y(self, data_processor):
        data_processor.train_dataset_scaled_ = self.scaler_X.fit_transform(data_processor.train_dataset_)
        data_processor.test_dataset_scaled_ = self.scaler_X.transform(data_processor.test_dataset_)
        data_processor.X_train_ = data_processor.train_dataset_scaled_[:, :-1]
        data_processor.X_test_ = data_processor.test_dataset_scaled_[:, :-1]
        data_processor.Y_train_ = data_processor.train_dataset_scaled_[:, -1]
        data_processor.Y_test_ = data_processor.test_dataset_scaled_[:, -1]

    def scale_data_without_Y(self, data_processor):
        data_processor.train_dataset_scaled_ = self.scaler_X.fit_transform(
            data_processor.train_dataset_.values[:, :-1])
        data_processor.test_dataset_scaled_ = self.scaler_X.transform(
            data_processor.test_dataset_.values[:, :-1])
        data_processor.X_train_ = data_processor.train_dataset_scaled_
        data_processor.X_test_ = data_processor.test_dataset_scaled_
        data_processor.Y_train_ = data_processor.train_dataset_.values[:, -1]
        data_processor.Y_test_ = data_processor.test_dataset_.values[:, -1]

    def scale_data_set_in_data_processor(self, data_processor):
        if self.config["should_scale_X"]:
            self.scale_X(data_processor)
        if self.config["should_scale_Y"]:
            self.scale_Y(data_processor)

    def scaler_transform(self, input_tab):
        out_tab = numpy.array(input_tab)
        if self.config["should_scale_X"]:
            out_tab[:-1] = self.scaler_X.transform([input_tab[:-1]])
        if self.config["should_scale_Y"]:
            out_tab[-1] = self.scaler_Y.transform([[input_tab[-1]]])
        return out_tab.tolist()

    def scaler_inverse_transform(self, input_tab):
        out_tab = numpy.array(input_tab)
        if self.config["should_scale_X"]:
            out_tab[:-1] = self.scaler_X.inverse_transform([input_tab[:-1]])
        if self.config["should_scale_Y"]:
            out_tab[-1] = self.scaler_Y.inverse_transform([[input_tab[-1]]])
        return out_tab.tolist()

    def scale_X(self, data_processor):
        data_processor.X_train_ = self.scaler_X.fit_transform(data_processor.X_train_)
        data_processor.X_test_ = self.scaler_X.transform(data_processor.X_test_)

    def scale_Y(self, data_processor):
        data_processor.Y_train_ = self.scaler_Y.fit_transform(data_processor.Y_train_[:, None])[:, 0]
        data_processor.Y_test_ = self.scaler_Y.transform(data_processor.Y_test_[:, None])[:, 0]
