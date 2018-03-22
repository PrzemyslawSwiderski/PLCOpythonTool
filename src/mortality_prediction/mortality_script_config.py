from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler

config = {
    "prediction_name": "Mortality",
    "average_life_duration": 77.0,
    "days_in_year": 365.242199,
    "boundary_values_to_exclude": [
        {"feature_name": "dth_days", "boundary_scale_value": 2.0},
        {"feature_name": "dx_psa", "boundary_scale_value": 2.0}
    ],
    "features_to_preserve_after_preprocessing": [
        "age",
        "dx_psa",
        "pros_gleason",
        "bmi_curc",
        "weight_f",
        "height_f",
        "rectal_history",
        "surg_age",
        "cig_years",
        "numbiopp",
        "curative_hormp",
        "curative_othp",
        "curative_prostp",
        "curative_radp",
        "asppd",
        "ibuppd",
        "dth_days"  # THE LAST PARAMETER IS AIMED TO BE PREDICTED
    ],
    "features_to_print_correlations": ["dth_days"],
    "scaler": StandardScaler()
    # "scaler": MinMaxScaler()
    # "scaler": MaxAbsScaler()
    # "scaler": Normalizer()
    # "scaler": RobustScaler()
}
