import numpy
import time

import sklearn.metrics as metrics
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.rod import ROD
from pyod.models.sampling import Sampling
from pyod.models.suod import SUOD
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from xgboost import XGBClassifier

from utils.Stacking import Stacker


def supervised_classifiers():
    class_list = []

    # Supervised
    class_list.append(["Logistic Regression,SUP", LogisticRegression(random_state=0)])
    class_list.append(["NaiveBayes,SUP", GaussianNB()])
    class_list.append(["LDA,SUP", LinearDiscriminantAnalysis()])
    class_list.append(["XGBoost Default,SUP", XGBClassifier()])
    class_list.append(["RandomForest 100,SUP", RandomForestClassifier(n_estimators=100, random_state=0)])
    class_list.append(["Decision Tree,SUP", DecisionTreeClassifier()])

    return class_list


def fast_unsupervised_classifiers(outliers_fraction):
    class_list = []

    # Unsupervised
    class_list.append(["COPOD,UNS", COPOD(contamination=outliers_fraction)])
    class_list.append(["ECOD,UNS", ECOD(contamination=outliers_fraction)])
    class_list.append(["MCD,UNS", MCD(contamination=outliers_fraction)])
    class_list.append(["PCA,UNS", PCA(contamination=outliers_fraction, weighted=True)])
    class_list.append(["CBLOF,UNS", CBLOF(contamination=outliers_fraction)])
    class_list.append(["ROD,UNS", ROD(contamination=outliers_fraction)])
    class_list.append(["iForest,UNS", IForest(contamination=outliers_fraction)])

    return class_list


def unsupervised_classifiers(outliers_fraction):
    class_list = []

    # Unsupervised
    class_list.append(["COPOD,UNS", COPOD(contamination=outliers_fraction)])
    class_list.append(["HBOS,UNS", HBOS(contamination=outliers_fraction)])
    class_list.append(["MCD,UNS", MCD(contamination=outliers_fraction)])
    class_list.append(["PCA,UNS", PCA(contamination=outliers_fraction, weighted=True)])
    class_list.append(["ECOD,UNS", ECOD(contamination=outliers_fraction)])
    class_list.append(["Sampling,UNS", Sampling(contamination=outliers_fraction)])
    class_list.append(["LOF,UNS", LOF(contamination=outliers_fraction)])
    class_list.append(["CBLOF,UNS", CBLOF(contamination=outliers_fraction)])
    class_list.append(["kNN,UNS", KNN(contamination=outliers_fraction)])
    class_list.append(["ROD,UNS", ROD(contamination=outliers_fraction)])
    class_list.append(["iForest,UNS", IForest(contamination=outliers_fraction)])
    class_list.append(["SUOD,UNS", SUOD(contamination=outliers_fraction, base_estimators=[COPOD(), PCA(), CBLOF()])])

    return class_list


def stacking_classifiers(base_level):
    class_list = []

    for [name, meta_level] in supervised_classifiers():
        class_list.append(["Stacking-" + name + "-noT,Custom",
                           Stacker(base_level_learners=base_level, meta_level_learner=meta_level, use_training=False)])
        class_list.append(["Stacking-" + name + ",Custom",
                           Stacker(base_level_learners=base_level, meta_level_learner=meta_level, use_training=True)])

    return class_list


def available_classifiers(outliers_fraction):
    class_list = []
    class_list.append(supervised_classifiers())
    class_list.append(stacking_classifiers(unsupervised_classifiers(outliers_fraction)))
    return class_list


def data_analysis(classifier, tr_x, tr_y, te_x, te_y, unk_x, unk_y):

    classifier_name = classifier[0]
    model = classifier[1]

    # Train
    start = time.time()
    model.fit(tr_x, tr_y)
    elapsed_train = (time.time() - start)/len(tr_y)

    # Scoring Test Confusion Matrix
    start = time.time()
    y_pred = model.predict(te_x)
    y_pred[y_pred < 0] = 0
    elapsed_test = (time.time() - start)/len(te_y)
    tn, fp, fn, tp = metrics.confusion_matrix(te_y, y_pred).ravel()
    accuracy = metrics.accuracy_score(te_y, y_pred)
    mcc = abs(metrics.matthews_corrcoef(te_y, y_pred))
    if accuracy < 0.5:
        accuracy = 1.0 - accuracy
        tp, fn = fn, tp
        tn, fp = fp, tn
    rec = tp / (tp + fn)

    # Scoring Test Unknown Matrix
    if len(unk_y) > 0:
        y_unk = model.predict(unk_x)
        y_unk[y_unk < 0] = 0
        tp_u = numpy.sum(unk_y == y_unk)
        fn_u = numpy.sum(unk_y != y_unk)
        rec_unk = tp_u / len(unk_y)
    else:
        tp_u = fn_u = rec_unk = 0
        y_unk = []

    print("Accuracy/MCC = " + '{0:.4f}'.format(accuracy) + "/" + '{0:.4f}'.format(mcc) +
          ", rec-unk = " + '{0:.4f}'.format(rec_unk) + " with " + str(len(unk_y)) +
          " (" + '{0:.4f}'.format(100*len(unk_y)/len(te_y)) + "%) unknowns [" + classifier_name + "] time " + str(elapsed_train) + " ms")

    return [classifier_name, elapsed_train, elapsed_test, tp, tn, fp, fn, accuracy, rec, mcc, tp_u, fn_u, rec_unk], y_pred, y_unk