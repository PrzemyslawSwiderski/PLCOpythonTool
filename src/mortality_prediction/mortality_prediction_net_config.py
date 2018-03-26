from sklearn.neural_network import MLPRegressor
from mortality_prediction.mortality_data_processor_config import config

config = {
    "test_input_data": [
        56.0,  # "age",
        4.70,  # "dx_psa",
        6.0,  # "pros_gleason",
        4.0,  # "bmi_curc",
        210.0,  # "weight_f",
        69.0,  # "height_f",
        0.0,  # "rectal_history",
        3.565378,  # "surg_age",
        41.0,  # "cig_years",
        1.0,  # "numbiopp",
        0.0,  # "curative_hormp",
        1.0,  # "curative_othp",
        0.0,  # "curative_prostp",
        0.0,  # "curative_radp",
        6.0,  # "asppd",
        0.0,  # "ibuppd",
        7670.086179,  # "dth_days"  # THE LAST PARAMETER IS AIMED TO BE PREDICTED
    ],
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
