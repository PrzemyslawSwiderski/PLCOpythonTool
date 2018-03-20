import glob
import logging
import os

import pandas


def ensure_dir(dir_name):
    """
    Function checks if there is a directory with specified dir_name
    if not, it is not existing, dir structure is created

    :param dir_name:
    """
    directory = os.path.dirname(dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_file_names_by_ext(path, extension='pkl'):
    """
    Method returns list of files with specified extension
    in directory

    :param path: directory
    :param extension: file_extension
    """
    files_grabbed = []
    files_grabbed.extend(glob.glob(path + os.sep + "*." + extension))
    return files_grabbed


def log_train_results_CV(rs, X_validation, Y_validation):
    logging.info("CV results:")
    logging.info(rs.cv_results_)
    logging.info("Best estimator:")
    logging.info(rs.best_estimator_)
    logging.info("Best params:")
    logging.info(rs.best_params_)
    logging.info("Real / Predicted values:")
    mlp_output_values_scaled = rs.predict(X_validation)
    logging.info(pandas.DataFrame({'predicted': mlp_output_values_scaled, 'real': Y_validation}))


def log_train_results_MLP(rs, X_validation, Y_validation):
    logging.info("Real / Predicted values:")
    mlp_output_values_scaled = rs.predict(X_validation)
    logging.info(
        pandas.DataFrame({'predicted': mlp_output_values_scaled, 'real': Y_validation}))
    logging.info("Score:")
    logging.info(rs.score(X_validation, Y_validation))
    logging.info("Loss:")
    logging.info(rs.loss_)


def get_redundant_pairs(data_frame):
    pairs_to_drop = set()
    cols = data_frame.columns
    for i in range(0, data_frame.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(data_frame, n=5):
    au_corr = data_frame.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data_frame)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
