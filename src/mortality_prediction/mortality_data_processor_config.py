from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler
from classes.data_processor_scaler import CommonDataProcessorScaler

config = {
    "prediction_name": "Mortality",
    "pickle_file_name": "predict_mortality.pkl",
    "query_file_name": "select_to_predict_mortality.sql",
    "average_life_duration": 77.0,
    "days_in_year": 365.242199,
    "boundary_values_to_exclude": [
        {"feature_name": "dth_days", "boundary_scale_value": 2.0},
        {"feature_name": "dx_psa", "boundary_scale_value": 2.0}
    ],
    "features_to_exclude_after_preprocessing": [
        "pros_exitage",
        # "age",
    ],
    "features_to_predict": [
        "dth_days",
    ],
    "features_to_print_correlations": ["dth_days", "pros_gleason", "curative_hormp", "curative_othp", "curative_prostp",
                                       "curative_radp"],
    "test_size": 0.1,
    # "scaler": CommonDataProcessorScaler({"transformer": MinMaxScaler(),
    #                                      "should_scale_Y": config["should_scale_Y"]}),
    "scaler": CommonDataProcessorScaler({
        "scaler_X": StandardScaler(),
        "scaler_Y": StandardScaler(),
        "should_scale_X": True,
        # "should_scale_X": True,
        "should_scale_Y": True,
        # "should_scale_Y": True
    }
    ),
    "random_state_split_value": 4,
    # "scaler": MinMaxScaler()
    # "scaler": MaxAbsScaler()
    # "scaler": Normalizer()
    # "scaler": RobustScaler()
    # "PCA_transform": {
    #     "PCA_object": PCA(n_components=4),
    # }
}
