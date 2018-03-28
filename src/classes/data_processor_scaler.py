class CommonDataProcessorScaler:

    def __init__(self, config):
        self.config = config
        self.scaler = self.config["transformer"]

    def scale_data_with_Y(self, data_processor):
        data_processor.train_dataset_scaled_ = self.scaler.fit_transform(data_processor.train_dataset_)
        data_processor.test_dataset_scaled_ = self.scaler.transform(data_processor.test_dataset_)
        data_processor.X_train_ = data_processor.train_dataset_scaled_[:, :-1]
        data_processor.X_test_ = data_processor.test_dataset_scaled_[:, :-1]
        data_processor.Y_train_ = data_processor.train_dataset_scaled_[:, -1]
        data_processor.Y_test_ = data_processor.test_dataset_scaled_[:, -1]

    def scale_data_without_Y(self, data_processor):
        data_processor.train_dataset_scaled_ = self.scaler.fit_transform(
            data_processor.train_dataset_.values[:, :-1])
        data_processor.test_dataset_scaled_ = self.scaler.transform(
            data_processor.test_dataset_.values[:, :-1])
        data_processor.X_train_ = data_processor.train_dataset_scaled_
        data_processor.X_test_ = data_processor.test_dataset_scaled_
        data_processor.Y_train_ = data_processor.train_dataset_.values[:, -1]
        data_processor.Y_test_ = data_processor.test_dataset_.values[:, -1]

    def scale_data_set_in_data_processor(self, data_processor):
        if self.config["should_scale_Y"]:
            self.scale_data_with_Y(data_processor)
        else:
            self.scale_data_without_Y(data_processor)

    def scaler_transform(self, input_tab):
        if 'transformer' in self.config: input_tab = self.scaler.transform(input_tab)
        return input_tab

    def scaler_inverse_transform(self, input_tab):
        if 'transformer' in self.config: input_tab = self.scaler.inverse_transform(input_tab)
        return input_tab
