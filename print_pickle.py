from sklearn.externals import joblib

train_cv_results = joblib.load('Output/train_cv_results.pkl')
train_best_estimator = joblib.load('Output/train_best_estimator.pkl')
train_best_params = joblib.load('Output/train_best_params.pkl')
print("CV results:")
print(train_cv_results)
print("Best estimator:")
print(train_best_estimator)
print("Best params:")
print(train_best_params)
