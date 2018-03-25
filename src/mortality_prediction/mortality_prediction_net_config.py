from sklearn.neural_network import MLPRegressor
from mortality_prediction.mortality_data_processor_config import config

config = {
    "test_input_data": [56.0, 4.70, 9.0, 7670.086179],
    "neural_net_def": MLPRegressor(activation='identity',
                                   solver='adam',
                                   hidden_layer_sizes=(15, 15),
                                   verbose=False,
                                   random_state=9,
                                   learning_rate='adaptive',
                                   momentum=0.09,
                                   max_iter=200,
                                   alpha=0.001),
    "should_scale_Y": config["should_scale_Y"],
    "saved_net_name": "saved_mortality_net.pkl"
}
