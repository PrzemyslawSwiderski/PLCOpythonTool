from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor

from mysql_fetcher import MySqlFetcher

ms = MySqlFetcher()

ms.run_select_query_from_file("queries/select_to_predict_mortality.sql")

ms.print_data_set_stats()
# train_data_set.hist()
# scatter_matrix(train_data_set)
# plt.show()
ms.print_scaled_data_set_stats()

X = ms.get_X()
Y = ms.get_Y()

validation_size = 0.1
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

mlp = MLPRegressor(activation='relu', solver='adam', hidden_layer_sizes=(15, 15), verbose=False, random_state=9,
                   max_iter=2000,
                   warm_start=False, shuffle=True, tol=0.0000001, learning_rate='adaptive', learning_rate_init=0.001)

rs = RandomizedSearchCV(mlp, param_distributions={
    'random_state': [1, 4, 7, 15],
    'solver': ['adam', 'lbfgs'],
    'hidden_layer_sizes': [(4, 15), (100, 100), (10, 10), (10, 10, 10)],
    'activation': ["identity", "relu"]}, verbose=10)
rs.fit(X_train, Y_train)

print("Real values:")
print(Y_validation)
print("Predicted values:")
mlp_output_values_scaled = rs.predict(X_validation)
print(mlp_output_values_scaled)

print("CV results:")
print(rs.cv_results_)
print("Best estimator:")
print(rs.best_estimator_)
print("Best params:")
print(rs.best_params_)

joblib.dump(rs.cv_results_, 'Output/train_cv_results.pkl')
joblib.dump(rs.best_estimator_, 'Output/train_best_estimator.pkl')
joblib.dump(rs.best_params_, 'Output/train_best_params.pkl')

# plt.plot(rs.best_estimator_.loss_curve)
# plt.title('MLPRegressor in scikit-learn loss curve')
# plt.show()

ms.close_connection()
