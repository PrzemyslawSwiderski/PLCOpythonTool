from sklearn.neural_network import MLPRegressor, MLPClassifier

config = {
    "test_input_data": [
        56.0,  # "age",
        4.70,  # "dx_psa",
        6.0,  # "pros_gleason",
        # 4.0,  # "bmi_curc",
        210.0,  # "weight_f",
        69.0,  # "height_f",
        0.0,  # "rectal_history",
        # 3.565378,  # "surg_age",
        41.0,  # "cig_years",
        1.0,  # "numbiopp",
        0.0,  # "curative_hormp",
        1.0,  # "curative_othp",
        0.0,  # "curative_prostp",
        0.0,  # "curative_radp",
        6.0,  # "asppd",
        0.0,  # "ibuppd",
    ],
    "neural_net_def": MLPClassifier(),
    "saved_net_name": "saved_treatment_net.pkl"
}
