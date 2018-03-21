import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neural_network import MLPRegressor

from classes.mysql_fetcher import MySqlFetcher

ms = MySqlFetcher()

ms.run_select_query_from_file("queries/select_to_predict_illness_probability.sql")

ms.print_data_set_stats()
# train_data_set.hist()
# scatter_matrix(train_data_set)
# plt.show()
ms.print_scaled_data_set_stats()

train_data_set_scaled = ms.get_scaled_data_set()
array = train_data_set_scaled.values

X = array[:, 0:2]
Y = array[:, 2]
validation_size = 0.15
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

mlp = MLPRegressor(activation='tanh', hidden_layer_sizes=(15, 10), verbose=True, random_state=1, max_iter=1000,
                   warm_start=True, shuffle=True, tol=0.00001, learning_rate='constant', learning_rate_init=0.001)

mlp.fit(X_train, Y_train)

print("Real values:")
print(Y_validation)
print("Predicted values:")
mlp_output_values_scaled = mlp.predict(X_validation)
print(mlp_output_values_scaled)

plt.plot(mlp.loss_curve_)
plt.title('MLPRegressor in scikit-learn loss curve')
plt.show()

ms.close_connection()
