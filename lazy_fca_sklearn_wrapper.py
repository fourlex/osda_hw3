from FCALC import fcalc
from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np


class MyBinarizedBinaryClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, support=None, method="standard", alpha=0):
        self.support = support
        self.method = method
        self.alpha = alpha

    def fit(self, X, y):
        assert X.dtype == bool
        assert y.dtype == bool
        # assert np.all(np.isclose(X, 0) | np.isclose(X, 1))
        # assert np.all(np.isclose(y, 0) | np.isclose(y, 1))
        # print(X)
        self.clf_ = fcalc.classifier.BinarizedBinaryClassifier(
            context=np.asarray(X, dtype=bool),
            labels=np.asarray(y, dtype=bool), 
            support=self.support,
            method=self.method,
            alpha=self.alpha,
        )
        self.classes_ = [0, 1]
        return self
    
    def predict(self, X):
        assert X.dtype == bool
        self.clf_.predict(np.asarray(X, dtype=bool))
        return self.clf_.predictions == 1


class MyPatternBinaryClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, support=None, method="standard", alpha=0, categorical=[]):
        self.support = support
        self.method = method
        self.alpha = alpha
        self.categorical = categorical

    def fit(self, X, y):
        # assert y.dtype == bool
        # print(X)
        self.clf_ = fcalc.classifier.PatternBinaryClassifier(
            context=np.asarray(X, dtype=float),
            labels=np.asarray(y, dtype=bool), 
            support=self.support,
            method=self.method,
            alpha=self.alpha,
            categorical=self.categorical
        )
        self.classes_ = [0, 1]
        return self
    
    def predict(self, X):
        # print(X)
        self.clf_.predict(np.asarray(X, dtype=float))
        return self.clf_.predictions == 1