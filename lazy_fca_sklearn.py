from FCALC import fcalc
from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np


class MyBinarizedBinaryClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, support=None, method="standard", alpha=0):
        self.support = support
        self.method = method
        self.alpha = alpha

    def fit(self, X, y):
        self.clf_ = fcalc.classifier.BinarizedBinaryClassifier(
            context=np.asarray(X, dtype=np.int8),
            labels=np.asarray(y, dtype=np.int8), 
            support=self.support,
            method=self.method,
            alpha=self.alpha,
        )
        self.classes_ = [0, 1]
        return self
    
    def predict(self, X):
        self.clf_.predict(np.asarray(X, dtype=np.int8))
        return self.clf_.predictions == 1
