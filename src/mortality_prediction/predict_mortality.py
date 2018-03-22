import argparse
import logging
import os

from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from common_config.common_config import LOGS_CATALOG_PATH, PICKLES_PATH
from common_config.logger_conf import configure_logging
from utils.helpers import log_train_results_MLPRegressor


def main():
    configure_logging(os.path.join(LOGS_CATALOG_PATH, "log_file.log"))
    parser = argparse.ArgumentParser(description="Script creates neural net model and predicts mortality")
    parser.add_argument("p", help="Pickled data_set file to be load")
    pickled_data_file = parser.parse_args().p

    data_set = joblib.load(os.path.join(PICKLES_PATH, f"saved_data_set_{pickled_data_file}.pkl"))

    logging.info("\n_________________________"
                 "\nInput DataSet Stats:"
                 f"\n{data_set.describe()}")

    # X = data_set.loc[:,("dx_psa", "pros_gleason", "numbiopp", "age", "bmi_curr", "cig_years", "rectal_history")]
    X = data_set.iloc[:, :-1]
    # X["pros_fh_age"].fillna(X["pros_fh_age"].mean(), inplace=True)
    Y = data_set.iloc[:, -1]

    validation_size = 0.1
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                        random_state=11)

    mlp = MLPRegressor(activation='identity', solver='adam', hidden_layer_sizes=(100, 100), verbose=False,
                       random_state=11,
                       max_iter=500,
                       alpha=0.001)

    rs = GridSearchCV(mlp, param_grid={
        'solver': ['adam'],
        'alpha': [0.001],
        'hidden_layer_sizes': [(100, 100)],
        'activation': ["identity"]}, verbose=10, cv=7)

    rs.fit(X_train, Y_train)

    log_train_results_MLPRegressor(rs.best_estimator_, X_test, Y_test)

    # log_train_results_MLP(mlp, X_train, Y_train)


if __name__ == "__main__":
    main()
