import logging

from pandas import DataFrame
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


def log_predict_result(target, predicted):
    abs_diff = abs(target - predicted)
    logging.info(f"\nPredicted value of dth_days: {predicted}")
    logging.info(f"\nTarget value of dth_days: {target}")
    logging.info(f"\nPredicted value and target abs diff: {abs_diff}")


def log_train_results_of_MLP(mlp, X_validation, Y_validation):
    mlp_predicted_values = mlp.predict(X_validation)
    fmt = '{} {}'
    logging.info(fmt.format('Real', 'Predicted'))
    for i, (real, predicted) in enumerate(zip(Y_validation.values, mlp_predicted_values)):
        logging.info(fmt.format(real, predicted))
        if i > 100: break
    logging.info(f"\nEstimator:\n{mlp}")
    logging.info(f"\nNumber of iterations: {mlp.n_iter_}")

    plt.plot(mlp.loss_curve_)
    plt.show()

    if type(mlp) is MLPRegressor:
        score = mlp.score(X_validation, Y_validation)
        logging.info("\nRegression coefficient of determination:"
                     f"\n{score}")
        logging.info("\nMean squared error:"
                     f"\n{mean_squared_error(Y_validation,mlp_predicted_values)}")
    else:
        logging.info("\nAll samples number:"
                     f"\n{Y_validation.shape[0]}")

        for i, predicted_feature in enumerate(list(Y_validation)):
            logging.info(f"\n_____________________________________________________")
            logging.info(f"\nPrediction results for a {predicted_feature} feature:")
            current_feature_Y_validation = Y_validation.values[:, i]
            current_feature_mlp_predicted_values = mlp_predicted_values[:, i]
            confusion_matrix_result = confusion_matrix(current_feature_Y_validation,
                                                       current_feature_mlp_predicted_values)
            logging.info(f"\nConfusion matrix:"
                         f"\n{confusion_matrix_result}")
            score = accuracy_score(current_feature_Y_validation, current_feature_mlp_predicted_values)
            logging.info("\nClassification Accuracy Score:"
                         f"\n{score}")
            count = accuracy_score(current_feature_Y_validation, current_feature_mlp_predicted_values, normalize=False)
            logging.info("\nClassification correct classified samples number:"
                         f"\n{count}")
