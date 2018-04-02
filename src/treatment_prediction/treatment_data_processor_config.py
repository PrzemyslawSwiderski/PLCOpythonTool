from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler
from classes.data_processor_scaler import CommonDataProcessorScaler

config = {
    "prediction_name": "Treatment",
    "pickle_file_name": "predict_treatment.pkl",
    "query_file_name": "select_to_predict_treatment.sql",
    "boundary_values_to_exclude": [
        {"feature_name": "dx_psa", "boundary_scale_value": 2.0}
    ],
    "features_to_exclude_after_preprocessing": [
        "pros_exitage",
        # "age",
    ],
    "features_to_predict": [
        # "curative_hormp",
        "curative_othp",
        "curative_prostp",
        # "curative_radp",
    ],
    "features_to_print_correlations": ["curative_hormp", "curative_othp", "curative_prostp",
                                       "curative_radp"],
    "validation_size": 0.1,
    # "scaler": CommonDataProcessorScaler({"transformer": MinMaxScaler(),
    #                                      "should_scale_Y": config["should_scale_Y"]}),
    "scaler": CommonDataProcessorScaler({
        "scaler_X": StandardScaler(),
        "scaler_Y": StandardScaler(),
        # "should_scale_X": False,
        "should_scale_X": True,
        # "should_scale_Y": False,
        "should_scale_Y": False
    }
    ),
    "random_state_split_value": 4,
    # "scaler": MinMaxScaler()
    # "scaler": MaxAbsScaler()
    # "scaler": Normalizer()
    # "scaler": RobustScaler()
    # "PCA_transform": {
    #     "PCA_object": PCA(n_components=7),
    # }
}
