import numpy as np
from sklearn.model_selection import train_test_split

X, Y = np.arange(20).reshape((5, 4)), range(5)
print("Zbiór X:")
print(X)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]
#  [16 17 18 19]]
print("Zbiór Y:")
print(list(Y))
# [0, 1, 2, 3, 4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=7)
print("Zbiór X_train:")
print(X_train)
# [[ 8  9 10 11]
#  [ 4  5  6  7]
#  [16 17 18 19]]
print("Zbiór X_test:")
print(X_test)
# [[ 0  1  2  3]
#  [12 13 14 15]]
print("Zbiór Y_train:")
print(list(Y_train))
# [2, 1, 4]
print("Zbiór Y_test:")
print(list(Y_test))
# [0, 3]
