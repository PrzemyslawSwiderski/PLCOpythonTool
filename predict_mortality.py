import argparse
import os

from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from config.config import PICKLES_PATH, LOGS_CATALOG_PATH
from config.logger_conf import configure_logging
from utils.helpers import log_train_results_MLP


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    parser = argparse.ArgumentParser(description="Script creates neural net model and predicts mortality")
    parser.add_argument("p", help="Pickled data_set file to be load")
    pickled_data_file = parser.parse_args().p

    data_set = joblib.load(os.path.join(PICKLES_PATH, f"{pickled_data_file}.pkl"))
    # X = data_set.loc[:,("dx_psa", "pros_gleason", "numbiopp", "age", "bmi_curr", "cig_years", "rectal_history")]
    X = data_set.loc[:, ("age", "cig_years", "pros_fh_age")]
    # X["pros_fh_age"].fillna(X["pros_fh_age"].mean(), inplace=True)
    Y = data_set.loc[:, "dth_days"]
    X.fillna(X.mean, inplace=True)
    validation_size = 0.1
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)

    mlp = MLPRegressor(activation='identity', solver='adam', hidden_layer_sizes=100, verbose=False, random_state=9,
                       max_iter=2000000,
                       warm_start=False, shuffle=True, tol=0.00000000001, learning_rate='adaptive',
                       learning_rate_init=0.001, alpha=100)
    # rs = GridSearchCV(mlp, param_grid={
    #     'alpha': [0.1],
    #     'hidden_layer_sizes': [(50, 50, 50)],
    #     'activation': ["identity"]}, verbose=10, n_jobs=4)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    mlp.fit(X_train, Y_train)
    log_train_results_MLP(mlp, X_train, Y_train)


if __name__ == "__main__":
    main()
