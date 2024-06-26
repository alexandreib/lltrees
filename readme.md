# Gradient Boost Tree in C++ 17:
	- Regression
	- Multiclasse Classification boost trees
	- Multiclasse Classification Adaboost trees.


### Python wrapper using Boost C++ libraries.
https://www.boost.org/

### dirs/files :
	MakeFile 
	cpp : sources of the gbt.
	demo : exemple jupyter notebook and python file to run a regression with the library.
	build : output of makefile (*.o, *.so).



# Build the C++ / .so library :
	!cd ../ && make clean
	!cd ../ && make

# Configuration / Usage :
## Configuration is done with a python Dictionnary :
```
conf ={
	'mode' : 'adaboost_classification',
	'epochs' : 2,
	'learning_rate' : 1,
	'metric' : 'accuracy',
	'max_depth' : 1,
	'min_leaf_size' : 1,
	'criterion' : "gini",  
	'verbose' : 1,
	'idx_cat_cols' : [8],
	'number_of_threads' : 8,
}
my_lltree.set_conf(conf)
```

mode : regression, classic_classification, adaboost_classification. <br/>
epochs : numbers of trees to fit. <br/>
learning_rate : self explaining. <br/>
metric : accuracy, mse, mae. limitation : can't use gini /Entropy criterion with Classic Boosted Tree. <br/>
max_depth : max depth of each tree.<br/>
min_leaf_size : minimum number of samples in each leaf. <br/>
criterion : criterion used of the split, possible values : variance, absolute_erorr, gini, entropy. <br/>
verbose : log at each epoch : 0 => no log. <br/>
idx_cat_cols : index of the categoricals columns in the datasets. <br/>
number_of_threads : numbers of thread used to fit the trees. <br/>



## Python logs examples :
FIT/PREDICT Log :
```-----------------------------------------
mode :              classic_classification
epochs :            2
learning_rate :     0.3
metric :            accuracy
criterion :         absolute_error
max_depth :         1
min_leaf_size :     1
idx_cat_cols :      [8, ]
verbose :           1
-----------------------------------------

Type of Training Data : int32
Configuration mode : classic_classification
Gbt_classic_classification fit
All the distinct element for classification in sorted order are: 0 1 
Epoch : 1     Metric Train : 0.82    Metric va : 0.826667 Residuals (sum) : 699.589
Epoch : 2     Metric Train : 0.82    Metric va : 0.826667 Residuals (sum) : 530.876

Print.
Tree : 0
└──1, criterion: 0.499853, size: 700
    ├──2, criterion: 0.188064, leaf_value: 1.6502, size: 276
    └──3, criterion: 0.352872, leaf_value: -1.01755, size: 424
Tree : 1
└──1, criterion: 0.499853, size: 700
    ├──2, criterion: 0.188064, leaf_value: -1.6502, size: 276
    └──3, criterion: 0.352872, leaf_value: 1.01755, size: 424
Tree : 2
└──1, criterion: 0.379293, size: 700
    ├──2, criterion: 0.220046, leaf_value: -1.03154, size: 349
    └──3, criterion: 0.245805, leaf_value: 1.1206, size: 351
Tree : 3
└──1, criterion: 0.379293, size: 700
    ├──2, criterion: 0.220046, leaf_value: 1.03154, size: 349
    └──3, criterion: 0.245805, leaf_value: -1.1206, size: 351
    
```
```
-----------------------------------------
mode :              regression
epochs :            5
learning_rate :     0.1
metric :            mae
criterion :         absolute_error
max_depth :         1
min_leaf_size :     2
idx_cat_cols :      [8, ]
verbose :           1
-----------------------------------------

Type of Training Data : float64
Configuration mode : regression
Epoch : 1     Metric Train : 93.5679 Metric va : 101.676 Residuals (sum) : -508.197
Epoch : 2     Metric Train : 90.6867 Metric va : 98.9202 Residuals (sum) : -457.378
Epoch : 3     Metric Train : 88.4611 Metric va : 96.1682 Residuals (sum) : -411.64
Epoch : 4     Metric Train : 86.1242 Metric va : 94.5949 Residuals (sum) : -370.476
Epoch : 5     Metric Train : 84.1233 Metric va : 92.273  Residuals (sum) : -333.428

Print.
Tree : 0
└──1, criterion: 95.8068, size: 700
    ├──2, criterion: 79.9472, leaf_value: -50.5767, size: 466
    └──3, criterion: 86.3583, leaf_value: 79.0032, size: 234
Tree : 1
└──1, criterion: 93.2837, size: 700
    ├──2, criterion: 76.6137, leaf_value: -86.3799, size: 266
    └──3, criterion: 81.5431, leaf_value: 42.4038, size: 434
Tree : 2
└──1, criterion: 90.4778, size: 700
    ├──2, criterion: 78.7713, leaf_value: -47.6568, size: 444
    └──3, criterion: 79.4921, leaf_value: 66.5751, size: 256
Tree : 3
└──1, criterion: 88.258, size: 700
    ├──2, criterion: 74.0077, leaf_value: -46.9349, size: 439
    └──3, criterion: 81.1795, leaf_value: 64.7497, size: 261
Tree : 4
└──1, criterion: 86.0214, size: 700
    ├──2, criterion: 74.8719, leaf_value: -56.9211, size: 339
    └──3, criterion: 77.6908, leaf_value: 44.216, size: 361
Save.
Print.
Tree : 0
└──1, criterion: 95.8068, size: 700
    ├──2, criterion: 79.9472, leaf_value: -50.5767, size: 466
    └──3, criterion: 86.3583, leaf_value: 79.0032, size: 234
Tree : 1
└──1, criterion: 93.2837, size: 700
    ├──2, criterion: 76.6137, leaf_value: -86.3799, size: 266
    └──3, criterion: 81.5431, leaf_value: 42.4038, size: 434
Tree : 2
└──1, criterion: 90.4778, size: 700
    ├──2, criterion: 78.7713, leaf_value: -47.6568, size: 444
    └──3, criterion: 79.4921, leaf_value: 66.5751, size: 256
Tree : 3
└──1, criterion: 88.258, size: 700
    ├──2, criterion: 74.0077, leaf_value: -46.9349, size: 439
    └──3, criterion: 81.1795, leaf_value: 64.7497, size: 261
Tree : 4
└──1, criterion: 86.0214, size: 700
    ├──2, criterion: 74.8719, leaf_value: -56.9211, size: 339
    └──3, criterion: 77.6908, leaf_value: 44.216, size: 361
```

# Current Limitations / to do list :
	- Save load working only for regression for the moment.
	- can't use gini /Entropy criterion with Classic Boosted Tree.
	- Need to had flexible path for save / load of the Trees.
	- features importances
	- add adaboost regression
	- add c++ dataframe integration ?
	https://github.com/hosseinmoein/DataFrame/tree/master
