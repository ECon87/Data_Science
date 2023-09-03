# Description: Call Lazypredict to select ML model
# Author: Evangelos Constantinou
#
# Variables to use in ML model -
# - Class_1, Class_2, Class_3
# - Embarked_NS
# - Female
# - ParCh_dm
# - Siblings
# - Log_Age
# - Log_Fare
# - Interactions:
#   - Class and Embarked_NS
#   - Class and Female (but lift is same direction for all groups - ignore)
#   - Class and Parents
#   - Class and Siblings
#   - Female and Parents
#   - Female and Siblings

import lazypredict
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import re
from lazypredict.Supervised import LazyClassifier
from seaborn.matrix import Grid
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from statsmodels.multivariate.factor import Model
from types import ModuleType

try:
    from gen_features import gen_features
except ModuleNotFoundError:
    from Code.gen_features import gen_features

# Directory
DIR_MASTER = "/home/econ87/Documents/Learning/Data_Science/Kaggle_Competitions/Titanic/"
DIR_DATA = f"{DIR_MASTER}/Data/"

# Load train data
data = pd.read_csv(f"{DIR_DATA}/train.csv", engine="pyarrow")

# Encode the outcome variable - SKIP since already encoded
# y_encoder = LabelEncoder()
# y_encoder.fit_transform(data.Survived)


# train-test
train_df, test_df = train_test_split(data, test_size=0.33)

# Features
train_df = gen_features(train_df)
test_df = gen_features(test_df)


features = [
    "Class_1",
    "Class_2",
    "Class_3",
    "Embarked_NS",
    "Female",
    "ParCh_dm",
    "Female_ParCh",
    "Siblings",
    "Female_Siblings",
    "Class_1_Embarked_NS",
    "Class_1_Female",
    "Class_1_ParCh",
    "Class_1_Siblings",
    "Class_2_Embarked_NS",
    "Class_2_Female",
    "Class_2_ParCh",
    "Class_2_Siblings",
    "Class_3_Embarked_NS",
    "Class_3_Female",
    "Class_3_ParCh",
    "Class_3_Siblings",
    "Log_Age",
    "Log_Fare",
]


X_train = train_df[features].values
y_train = train_df["Survived"].values

X_test = test_df[features].values
y_test = test_df["Survived"].values


# ==================================================================
# Feature selection - SKIP since we don't have a a lot features
# ==================================================================

# Correlation between features
pure_features = [i for i in features if bool(re.search(r"Class_\d_", i)) == False]
pure_features.remove("Female_ParCh")
pure_features.remove("Female_Siblings")

# corr_heatmap = sb.heatmap(train_df[pure_features].corr(), cmap="YlGnBu", annot=True)
colors = sb.dark_palette("#69d", reverse=True, as_cmap=True)
corr_heatmap = sb.heatmap(train_df[pure_features].corr(), cmap=colors, annot=True)
mp.show()

# SelectKBest
selector = SelectKBest(k=15)
X_train_new = selector.fit_transform(X_train, y_train)
X_train.shape
X_train_new.shape

X_test_new = selector.transform(X_test)

# ==================================================================
# Invoke Lazypredict to get all models
# ==================================================================
clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
# models, predictions = clf.fit(X_train_new, X_test_new, y_train, y_test)

print(models)


# ==================================================================
# Optimize XGBoost model
# GridSearchCV: best model is learning 
# - Best model: {'n_estimators': 25, learning_rate: '0.1'}

parameters = {'n_estimators': [10, 25, 50, 100, 150, 200],
              'learning_rate' : [0.025, 0.05, 0.075, 0.1],
              # 'eval_metric': ['auc']
              }
clf = GridSearchCV(XGBClassifier(), parameters, cv = 5)
clf.fit(X_train, y_train)
clf.cv_results_.keys()
len(clf.cv_results_['params'])
clf.cv_results_['mean_test_score']
clf.cv_results_['std_test_score']
print(f'Best model according to GridSearchCV: {clf.best_params_}, {clf.best_score_}')

for e, i in enumerate(clf.cv_results_['params']):
    print(i['learning_rate'], i['n_estimators'],
          round(clf.cv_results_['mean_test_score'][e], 3),
          round(clf.cv_results_['std_test_score'][e], 3),
          )

# Generate predictions based on best_params_
clf_preds = clf.best_estimator_.predict(X_test)
print(accuracy_score(y_test, clf_preds))


# Do cross validation between the model suggested by GridSearch and the one I
# found to perform very well

models = {
        'model1': {'n_estimators': 50, 'learning_rate': 0.1},
        'model2': {'n_estimators': 100, 'learning_rate': 0.1},
        'model3': {'n_estimators': 150, 'learning_rate': 0.025},
        }


# Model 1 (GridSearchCV)
clf_1 = cross_val_score(XGBClassifier(**models['model1']), X_train, y_train, cv = 10)

# Model 2 (Own model)
clf_2 = cross_val_score(XGBClassifier(**models['model2']), X_train, y_train, cv = 10)

# Model 3 (Own model)
clf_3 = cross_val_score(XGBClassifier(**models['model3']), X_train, y_train, cv = 10)

print(f'Relative average score of the two models {clf_1.mean()/clf_2.mean()}')
print(f'Relative std of the two models {clf_1.std()/clf_2.std()}')
print(f'Relative average score of the two models {clf_1.mean()/clf_3.mean()}')
print(f'Relative std of the two models {clf_1.std()/clf_3.std()}')

bst = XGBClassifier(**models['model3'])
bst.fit(X_train, y_train)
preds = bst.predict(X_test)

print(confusion_matrix(y_test, preds))
print(accuracy_score(y_test, preds))
print(f1_score(y_test, preds))

