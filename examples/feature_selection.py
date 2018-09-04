import numpy
import pandas
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.linear_model import LassoCV
iris = load_iris()
iris = pandas.DataFrame(data=numpy.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
X, y = iris.values[:, :-1], iris.target
print("Cechy przed filtrowaniem:")
print(list(iris.iloc[:, :-1]))
# Output:
# Cechy przed filtrowaniem:
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Wybór dwóch najlepszych cech według testu chi-kwadrat
selector = SelectKBest(chi2, k=2)
selector.fit(X, y)
# Get idxs of columns to keep
idxs_selected = selector.get_support(indices=True)
features_dataframe_new = iris.iloc[:, idxs_selected]
print("Cechy po filtrowaniu poprzez test chi-kwadrat:")
print(list(features_dataframe_new))
# Output:
# Cechy po filtrowaniu poprzez test chi-kwadrat:
# ['petal length (cm)', 'petal width (cm)']

clf = LassoCV()
# Odrzucenie cech, których "przydatność" jest mniejsza niż 0.25
selector = SelectFromModel(clf, threshold=0.25)
selector.fit(X, y)
idxs_selected = selector.get_support(indices=True)
features_dataframe_new = iris.iloc[:, idxs_selected]
print("Cechy po filtrowaniu poprzez klasyfikator LassoCV:")
print(list(features_dataframe_new))
# Output:
# Cechy po filtrowaniu poprzez klasyfikator LassoCV:
# ['petal length (cm)']
