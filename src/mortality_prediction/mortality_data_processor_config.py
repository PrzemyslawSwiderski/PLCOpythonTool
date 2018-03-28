from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler
from mortality_prediction.mortality_prediction_net_config import config
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
    ],
    "features_to_print_correlations": ["dth_days"],
    "validation_size": 0.1,
    # "scaler": MinMaxScaler(),
    "scaler": CommonDataProcessorScaler({"transformer": StandardScaler(),
                                         "should_scale_Y": config["should_scale_Y"]}),
    "random_state_split_value": 9,
    # "scaler": MinMaxScaler()
    # "scaler": MaxAbsScaler()
    # "scaler": Normalizer()
    # "scaler": RobustScaler()
    # "PCA_transform": {
    #     "PCA_object": PCA(n_components=13, svd_solver='arpack'),
    # }
}
