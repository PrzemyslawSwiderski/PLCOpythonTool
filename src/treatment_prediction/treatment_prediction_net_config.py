import pandas
from sklearn.neural_network import MLPRegressor, MLPClassifier

config = {
    "test_input_data": pandas.DataFrame(data={
        "age": [56.0],
        "dx_psa": [4.70],
        "pros_gleason": [6.0],
        "weight_f": [210.0],
        "height_f": [69.0],
        "cig_years": [41.0],
        "numbiopp": [1.0],
        "curative_hormp": [0.0],
        "curative_prostp": [0.0],
        "curative_radp": [0.0],
        "asppd": [6.0],
        "ibuppd": [0.0],
        "curative_othp": [1.0]  # THE LAST PARAMETER IS AIMED TO BE PREDICTED
    }),
    "neural_net_def": MLPClassifier(activation='logistic',
                                    solver='adam',
                                    hidden_layer_sizes=(15, 15),
                                    verbose=False,
                                    random_state=7,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=0.02,
                                    learning_rate='adaptive',
                                    max_iter=200,
                                    alpha=0.001),
    "saved_net_name": "saved_treatment_net.pkl"
}
