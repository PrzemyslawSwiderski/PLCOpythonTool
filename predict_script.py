from datetime import datetime
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from helpers import log_train_results
from mysql_fetcher import MySqlFetcher
import logging
import logging.config


def main():
    logging.config.fileConfig("logging.conf")

    ms = MySqlFetcher()
    ms.run_select_query_from_file("queries/select_to_predict_mortality.sql")
    ms.details_printer.print_all()
    X = ms.get_X()
    Y = ms.get_Y()
    validation_size = 0.1
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)
    mlp = MLPRegressor(activation='relu', solver='lbfgs', hidden_layer_sizes=(15, 15), verbose=False, random_state=9,
                       max_iter=20000,
                       warm_start=False, shuffle=True, tol=0.000000001, learning_rate='adaptive',
                       learning_rate_init=0.001)
    rs = GridSearchCV(mlp, param_grid={
        'alpha': [0.1, 1000],
        'hidden_layer_sizes': [(100, 100, 100), (50, 50)],
        'shuffle': [True, False],
        'activation': ["identity", "tanh"]}, verbose=10, n_jobs=4)
    rs.fit(X_train, Y_train)
    log_train_results(rs, X_validation, Y_validation)
    training_result = {
        "result_set": rs,
        "input_data_set": ms.data_set,
        "input_data_set_normalized": ms.data_set_normalized,
        "train_test_split": (X_train, X_validation, Y_train, Y_validation)
    }
    date_time_now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    joblib.dump(training_result, f'pickles/train_result_{date_time_now}.pkl')
    # plt.plot(rs.best_estimator_.loss_curve)
    # plt.title('MLPRegressor in scikit-learn loss curve')
    # plt.show()
    ms.close_connection()


if __name__ == "__main__":
    main()
