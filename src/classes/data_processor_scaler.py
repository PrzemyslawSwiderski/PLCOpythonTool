from pandas import DataFrame

from utils.helpers import fit_transform_data_frame, transform_data_frame, exclude_data_frame_columns, \
    inverse_transform_data_frame


class CommonDataProcessorScaler:
    def __init__(self, config):
        self.config = config
        self.scaler_X = self.config["scaler_X"]
        self.scaler_Y = self.config["scaler_Y"]

    def scale_data_set_in_data_processor(self, data_processor):
        if self.config["should_scale_X"]:
            self.scale_X(data_processor)
        if self.config["should_scale_Y"]:
            self.scale_Y(data_processor)
        pass

    def scaler_transform(self, input_tab, features_to_predict):
        out_tab = DataFrame(input_tab)
        if self.config["should_scale_X"]:
            out_tab = transform_data_frame(self.scaler_X, exclude_data_frame_columns(input_tab, features_to_predict))
            out_tab[features_to_predict] = input_tab[features_to_predict]
        if self.config["should_scale_Y"]:
            out_tab[features_to_predict] = transform_data_frame(self.scaler_Y, input_tab[
                features_to_predict])
        return out_tab

    def scaler_inverse_transform(self, input_tab, features_to_predict):
        out_tab = DataFrame(input_tab)
        if self.config["should_scale_X"]:
            out_tab = inverse_transform_data_frame(self.scaler_X,
                                                   exclude_data_frame_columns(input_tab, features_to_predict))
            out_tab[features_to_predict] = input_tab[features_to_predict]
        if self.config["should_scale_Y"]:
            out_tab[features_to_predict] = inverse_transform_data_frame(self.scaler_Y, input_tab[
                features_to_predict])
        return out_tab

    def scale_X(self, data_processor):
        data_processor.X_train_ = fit_transform_data_frame(self.scaler_X, data_processor.X_train_)
        data_processor.X_test_ = transform_data_frame(self.scaler_X, data_processor.X_test_)

    def scale_Y(self, data_processor):
        data_processor.Y_train_ = fit_transform_data_frame(self.scaler_Y, data_processor.Y_train_)
        data_processor.Y_test_ = transform_data_frame(self.scaler_Y, data_processor.Y_test_)
