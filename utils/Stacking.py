import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class Stacker(BaseEstimator, ClassifierMixin):

    def __init__(self, base_level_learners, meta_level_learner, use_training=False):
        self.base_level_learners = base_level_learners
        self.meta_level_learner = meta_level_learner
        self.use_training = use_training

    def fit(self, X, y):
        # SKLearn-friendly actions
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        stacking_data = []

        # Training base-level-learners
        for name, learner in self.base_level_learners:
            learner.fit(X, y)
            stacking_data.append(learner.predict_proba(X))

        stacking_data = numpy.concatenate(stacking_data, axis=1)
        if self.use_training:
            stacking_data = numpy.concatenate([X, stacking_data], axis=1)

        # Trains meta-level learner
        self.meta_level_learner.fit(stacking_data, y)

        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        stacking_data = []

        # Training base-level-learners
        for name, learner in self.base_level_learners:
            stacking_data.append(learner.predict_proba(X))

        stacking_data = numpy.concatenate(stacking_data, axis=1)
        if self.use_training:
            stacking_data = numpy.concatenate([X, stacking_data], axis=1)

        return self.meta_level_learner.predict(stacking_data)
