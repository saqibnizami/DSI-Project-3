from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd

# build pipelines for RandomForest and Logistic Regression
rf_pipe = Pipeline([('select', SelectPercentile(score_func=chi2,
                                            percentile=10)),
                ('classifier', RandomForestClassifier())])

logreg_pipe = Pipeline([('select', SelectPercentile(score_func=chi2,
                                                percentile=10)),
                    ('classifier', LogisticRegression())])


# set percentiles for feature selection
percentiles = [10, 20, 30, 40]


# build parameter grids for Random Forest and Logistic Regression
rf_param_grid = {'classifier__max_depth': [3, None],
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__class_weight': ['balanced', 'balanced_subsample'],
                'select': percentiles}

logreg_param_grid = {'classifier__solver': ['liblinear'],
                 'classifier__C': (0.0001, 0.01, 0.1, 1.0, 10, 10000),
                 'classifier__penalty': ['l1', 'l2'],
                 'select': percentiles}


# use exhaustive grid search to find best hyperparameters and features
rf_search = GridSearchCV(rf_pipe, param_grid=rf_param_grid, cv=10, scoring='roc_auc',
                         n_jobs=-1).fit(features_train, target_train)

logreg_search = GridSearchCV(logreg_pipe, param_grid=logreg_param_grid, cv=10, scoring='roc_auc',
                             n_jobs=-1).fit(features_train, target_train)