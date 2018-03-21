from sklearn.externals import joblib

from helpers import get_file_names_by_ext

pickles = get_file_names_by_ext("pickles")
pickles_sorted = sorted(pickles)
rs = joblib.load(pickles_sorted[-1])

print("Best estimator:")
print(rs.best_estimator_)
print("Best params:")
print(rs.best_params_)
