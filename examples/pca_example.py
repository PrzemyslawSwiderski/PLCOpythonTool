import numpy
import pandas
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris = load_iris()
iris = pandas.DataFrame(data=numpy.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
X, y = iris.values[:, :-1], iris.target
print("Liczba cech przed dekompozycją:")
print(X.shape[1])
# Output:
# Liczba cech przed dekompozycją:
# 4
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
print("Procent wariancji wyjaśniony przez każdy z wybranych składników:")
print(pca.explained_variance_ratio_)
# Output:
# Procent wariancji wyjaśniony przez każdy z wybranych składników:
# [ 0.92461621  0.05301557]
print("Wartości osobliwe dla każdego z wybranych składników:")
print(pca.singular_values_)
# Output:
# Wartości osobliwe dla każdego z wybranych składników:
# [ 25.08986398   6.00785254]
X_new = pca.fit_transform(X)
print("Liczba cech po wykonaniu przekształcenia PCA:")
print(X_new.shape[1])
# Output:
# Liczba cech po wykonaniu przekształcenia PCA:
# 2
