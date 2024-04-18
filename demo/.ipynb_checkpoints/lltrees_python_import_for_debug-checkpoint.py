import time
import numpy as np
import pandas as pd
import time
import sklearn.datasets, sklearn.metrics, sklearn.model_selection, sklearn.tree
from sklearn.datasets import load_breast_cancer

import sys
sys.path.append('/home/alexandre/Desktop/lltrees/build')

X0, Y0 = sklearn.datasets.make_classification(n_samples=1000, n_features=8, n_informative=5, n_classes=2, random_state=40)
X1, Y1 = sklearn.datasets.make_classification(n_samples=1000, n_features=8, n_informative=5, n_classes=2, random_state=42)

zeros = np.full((len(Y0), 1), 0)
ones = np.full((len(Y1), 1), 1)
    
X0 = np.hstack((X0, zeros))
X1 = np.hstack((X1, ones))

X = np.vstack((X0, X1))
Y = np.hstack((Y0, Y1))
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)

import lltrees

my_lltree = lltrees.lltree()

conf ={
    'mode' : 'classic_classification',
    'epochs' : 2,
    'learning_rate' : 0.3,
    'metric' : 'accuracy',
    'max_depth' : 1,
    'min_leaf_size' : 1,
    'criterion' : "absolute_error",  
    'verbose' : 1,  
    'idx_cat_cols' : [8],
}

my_lltree.set_conf(conf)
my_lltree.get_conf()

start_time = time.time()
my_lltree.fit(X_train, Y_train.astype(np.int32), X_test, Y_test.astype(np.int32))
print("FIT --- %s seconds ---" % (time.time() - start_time))

my_lltree.print()

start_time = time.time()
YP = my_lltree.predict_proba(X_test)
print("PREDICT PROBA --- %s seconds ---" % (time.time() - start_time))
df = pd.DataFrame(YP)
print(df.shape)
print(YP[:10])

start_time = time.time()
YP = my_lltree.predict(X_test)
print("PREDICT --- %s seconds ---" % (time.time() - start_time))
df = pd.DataFrame(YP)
print(df.shape)
print("accuracy_score: %.2f" % np.sqrt(sklearn.metrics.accuracy_score(Y_test,YP)))
print(np.unique(YP, return_counts=True))

my_lltree.save()
my_lltree.load()
my_lltree.get_conf()


# start_time = time.time()
# YP = my_lltree.predict(X_test)
# print("PREDICT --- %s seconds ---" % (time.time() - start_time))
# df = pd.DataFrame(YP)
# print(df.shape)

# print("accuracy_score: %.2f" % np.sqrt(sklearn.metrics.accuracy_score(Y_test,YP)))
# print(np.unique(YP, return_counts=True))
del my_lltree

time.sleep(0.5)

X0, Y0 = sklearn.datasets.make_regression(n_samples=1000, n_features=9, n_informative=5, n_targets=1, noise=1, random_state=40)
X1, Y1 = sklearn.datasets.make_regression(n_samples=1000, n_features=9, n_informative=5, n_targets=1, noise=1, random_state=42)

zeros = np.full((len(Y0), 1), 0)
ones = np.full((len(Y1), 1), 1)
    
X0 = np.hstack((X0, zeros))
X1 = np.hstack((X1, ones))

X = np.vstack((X0, X1))
Y = np.hstack((Y0, Y1))
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)

conf ={
    'mode' : 'regression',
    'epochs' : 5,
    'learning_rate' : 0.1,
    'max_depth' : 1,
    'min_leaf_size' : 2,
    'verbose' : 1, 
    'metric' : 'mae', 
    'criterion' : "absolute_error", 
    'idx_cat_cols' : [8],
}
my_lltree = lltrees.lltree()
my_lltree.set_conf(conf)
my_lltree.get_conf()

start_time = time.time()
my_lltree.fit(X_train, Y_train, X_test, Y_test)
print("FIT --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

YP = my_lltree.predict(X_test)
print("PREDICT --- %s seconds ---" % (time.time() - start_time))
print(my_lltree.get_residuals())

my_lltree.print()
print(YP[:10])

print("rmse: %.2f" % np.sqrt(sklearn.metrics.mean_squared_error(Y_test,YP)))
print("mae: %.2f" % sklearn.metrics.mean_absolute_error(Y_test,YP))
print("r2: %.2f" % sklearn.metrics.r2_score(Y_test,YP))

my_lltree.save()
my_lltree.print()
del my_lltree


# time.sleep(0.5)
# X, Y = sklearn.datasets.make_classification(n_samples=1000, n_features=8, n_informative=5, n_classes=2, random_state=40)
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)

# my_lltree = lltrees.lltree()
# my_lltree.set_conf(conf)
# my_lltree.get_conf()
# my_lltree.load()
# my_lltree.print()

# # start_time = time.time()
# # my_lltree.fit(X_train, Y_train, X_test, Y_test)
# # print("FIT --- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# YP = my_lltree.predict(X_test)
# print("PREDICT --- %s seconds ---" % (time.time() - start_time))
# print(my_lltree.get_residuals())

# print("rmse: %.2f" % np.sqrt(sklearn.metrics.mean_squared_error(Y_test,YP)))
# print("mae: %.2f" % sklearn.metrics.mean_absolute_error(Y_test,YP))
# print("r2: %.2f" % sklearn.metrics.r2_score(Y_test,YP))
# print(YP[0:10])


# del my_lltree
# my_lltree = lltrees.lltree()

# conf ={
#     'mode' : 'adaboost_classification',
#     'epochs' : 15,
#     'learning_rate' : 1,
#     'metric' : 'accuracy',
#     'max_depth' : 1,
#     'min_leaf_size' : 1,
#     'criterion' : "gini",  
#     'verbose' : 1,  
# }

# my_lltree.set_conf(conf)
# my_lltree.get_conf()

# start_time = time.time()
# my_lltree.fit(X_train, Y_train.astype(np.int32), X_test, Y_test.astype(np.int32))
# print("FIT --- %s seconds ---" % (time.time() - start_time))

# my_lltree.print()

# YP = my_lltree.predict_proba(X_test)
# df = pd.DataFrame(YP)
# print(df.shape)
# print(YP[:10])


# start_time = time.time()
# YP = my_lltree.predict(X_test)
# df = pd.DataFrame(YP)
# print(df.shape)
# print(YP[:10])

# print("PREDICT --- %s seconds ---" % (time.time() - start_time))

# print("accuracy_score: %.2f" % np.sqrt(sklearn.metrics.accuracy_score(Y_test,YP)))
# print(np.unique(YP, return_counts=True))
# del my_lltree