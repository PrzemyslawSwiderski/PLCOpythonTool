from sklearn import preprocessing
import numpy as np

X = np.array([[2.0, -2.0, 3.0],
              [2.0, -1.2, 4.4],
              [0.5, 0.0, 1.0]])

X_scaled_standard = preprocessing.StandardScaler().fit_transform(X)
print("X_scaled_standard by StandardScaler:")
print(X_scaled_standard)
# [[ 0.70710678 -1.13554995  0.14334554]
#  [ 0.70710678 -0.16222142  1.14676436]
#  [-1.41421356  1.29777137 -1.2901099 ]]
print("X_scaled_standard mean values for columns:")
print(X_scaled_standard.mean(axis=0))
# [  0.00000000e+00   0.00000000e+00  -7.40148683e-17]
print("X_scaled_standard unit variances values for columns:")
print(X_scaled_standard.std(axis=0))
# [ 1.  1.  1.]
print("X_scaled_standard max value:")
print(X_scaled_standard.max())
# 1.29777136905
print("X_scaled_standard min value:")
print(X_scaled_standard.min())
# -1.41421356237
X_scaled_minMax = preprocessing.MinMaxScaler().fit_transform(X)
print("X_scaled_minMax by MinMaxScaler:")
print(X_scaled_minMax)
# [[ 1.          0.          0.58823529]
#  [ 1.          0.4         1.        ]
#  [ 0.          1.          0.        ]]
print("X_scaled_minMax mean values for columns:")
print(X_scaled_minMax.mean(axis=0))
# [ 0.66666667  0.46666667  0.52941176]
print("X_scaled_minMax unit variances values for columns:")
print(X_scaled_minMax.std(axis=0))
# [ 0.47140452  0.41096093  0.41036176]
print("X_scaled_minMax max value:")
print(X_scaled_minMax.max())
# 1.0
print("X_scaled_minMax min value:")
print(X_scaled_minMax.min())
# 0.0
