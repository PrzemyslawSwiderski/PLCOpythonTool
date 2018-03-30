import logging

from pandas import DataFrame
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neural_network import MLPRegressor


def log_predict_result(target, predicted):
    abs_diff = abs(target - predicted)
    logging.info(f"\nPredicted value of dth_days: {predicted}")
    logging.info(f"\nTarget value of dth_days: {target}")
    logging.info(f"\nPredicted value and target abs diff: {abs_diff}")


def log_train_results_of_MLP(mlp, X_validation, Y_validation):
    logging.info("\nReal / Predicted values:\n")
    mlp_predicted_values = mlp.predict(X_validation)
    results = DataFrame()
    results["predicted"] = mlp_predicted_values
    results["target"] = Y_validation.values
    logging.info(results)
    logging.info(f"\nEstimator:\n{mlp}")
    logging.info(f"\nNumber of iterations: {mlp.n_iter_}")

    if type(mlp) is MLPRegressor:
        score = mlp.score(X_validation, results.target.values)
    else:
        accuracy_score(results.target.values, results.predicted.values)
    logging.info("\nAccuracy Score:"
                 f"\n{score}")
    logging.info("\nMean squared error:"
                 f"\n{mean_squared_error(results.target.values,results.predicted.values)}")
