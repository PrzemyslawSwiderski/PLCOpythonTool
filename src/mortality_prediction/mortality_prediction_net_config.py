import pandas
from sklearn.neural_network import MLPRegressor

config = {
    "test_input_data": pandas.DataFrame(data={
        "age": [56.0],
        "dx_psa": [4.70],
        "pros_gleason": [6.0],
        # "bmi_curc": [4.0],
        "weight_f": [210.0],
        "height_f": [69.0],
        "rectal_history": [0.0],
        # "surg_age": [3.565378],
        "cig_years": [41.0],
        "numbiopp": [1.0],
        "curative_hormp": [0.0],
        "curative_othp": [1.0],
        "curative_prostp": [0.0],
        "curative_radp": [0.0],
        "asppd": [6.0],
        "ibuppd": [0.0],
        "dth_days": [7670.086179]  # THE LAST PARAMETER IS AIMED TO BE PREDICTED
    }),
    "neural_net_def": MLPRegressor(activation='relu',
                                   solver='adam',
                                   hidden_layer_sizes=(15, 15),
                                   verbose=False,
                                   random_state=9,
                                   learning_rate='adaptive',
                                   momentum=0.09,
                                   max_iter=1000,
                                   alpha=0.001),
    "neural_net_def_bak": MLPRegressor(activation='identity',
                                       solver='adam',
                                       hidden_layer_sizes=(15, 15),
                                       verbose=False,
                                       random_state=9,
                                       learning_rate='adaptive',
                                       momentum=0.09,
                                       max_iter=200,
                                       alpha=0.001),
    "saved_net_name": "saved_mortality_net.pkl"
}
