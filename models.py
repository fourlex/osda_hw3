from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


models_and_param_grids = {
    'nb': (BernoulliNB(binarize=False), {
        'bernoullinb__alpha': [1.0,]
    }),
    'xgb': (XGBClassifier(random_state=0, objective='binary:logistic'), {
        'xgbclassifier__n_estimators': [100,],
    }),
    'rf': (RandomForestClassifier(random_state=0), {
        'randomforestclassifier__n_estimators': [10,],
    }),
    'logreg': (LogisticRegression(random_state=0), {
        'logisticregression__C': [1,], 
    }),
    'knn': (KNeighborsClassifier(), {
        'kneighborsclassifier__n_neighbors': [5,],
        'kneighborsclassifier__weights': ['distance'],
        # 'kneighborsclassifier__metric': ['hamming'],
    }),
    'catboost': (CatBoostClassifier(verbose=0, random_state=0), {
        'catboostclassifier__n_estimators': [100,],
    }),
}
