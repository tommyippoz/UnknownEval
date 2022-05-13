import numpy
import numpy as np
import glob
import time

import sklearn.metrics as metrics
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cd import CD
from pyod.models.cof import COF
from pyod.models.iforest import IForest
from pyod.models.kde import KDE
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.rod import ROD
from pyod.models.sampling import Sampling
from pyod.models.sod import SOD
from pyod.models.suod import SUOD
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest, BaggingClassifier, HistGradientBoostingClassifier, \
    StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from xgboost import XGBClassifier

from utils.PYODEstimator import PYODEstimator
from utils.dataset_utils import load_tabular_dataset

DATASETS_DIR = "datasets"
SCORES_FILENAME = "unk_scores.csv"

NORMAL_TAG = 'normal'
LABEL_NAME = 'multilabel'

TRAIN_VALIDATION_SPLIT = 0.5


def available_classifiers(outliers_fraction):
    class_list = []

    estimators = [
        ('ecod', PYODEstimator(estimator=ECOD(contamination=outliers_fraction))),
        ('copod', PYODEstimator(estimator=COPOD(contamination=outliers_fraction)))]
    class_list.append(["Stacking,UNS", StackingClassifier(estimators=estimators, final_estimator=XGBClassifier())])
    class_list.append(["XGBoost Default,SUP", XGBClassifier()])

    class_list.append(["ECOD,UNS", ECOD(contamination=outliers_fraction)])
    # class_list.append(["test,UNS", KDE(contamination=outliers_fraction)]) LENTO
    class_list.append(["Sampling,UNS", Sampling(contamination=outliers_fraction)])
    #class_list.append(["test,UNS", CD(contamination=outliers_fraction)]) CRASHA
    #class_list.append(["test,UNS", LMDD(contamination=outliers_fraction)]) LENTO
    class_list.append(["LOF,UNS", LOF(contamination=outliers_fraction)])
    #class_list.append(["test,UNS", COF(contamination=outliers_fraction)]) LENTO
    class_list.append(["CBLOF,UNS", CBLOF(contamination=outliers_fraction)])
    #class_list.append(["test,UNS", LOCI(contamination=outliers_fraction)]) LENTO
    class_list.append(["kNN,UNS", KNN(contamination=outliers_fraction)])
    #class_list.append(["test,UNS", SOD(contamination=outliers_fraction)]) CIUCCIA RISORSE
    class_list.append(["ROD,UNS", ROD(contamination=outliers_fraction)])
    #class_list.append(["test,UNS", OCSVM(contamination=outliers_fraction)])
    class_list.append(["iForest,UNS", IForest(contamination=outliers_fraction)])
    class_list.append(["FB,UNS", FeatureBagging(contamination=outliers_fraction)])
    #class_list.append(["test,UNS", LSCP(contamination=outliers_fraction, detector_list=[COPOD(), ECOD()])]) LENTO
    #class_list.append(["test,UNS", LODA(contamination=outliers_fraction)]) LENTO
    class_list.append(["test,UNS", SUOD(contamination=outliers_fraction, base_estimators=[COPOD(), ECOD()])])

    class_list.append(["test,SUP", PYODEstimator(estimator=COPOD(contamination=outliers_fraction))])

    # Supervised
    class_list.append(["Logistic Regression,SUP", LogisticRegression(random_state=0)])
    class_list.append(["NaiveBayes,SUP", GaussianNB()])
    class_list.append(["LDA,SUP", LinearDiscriminantAnalysis()])
    class_list.append(["XGBoost Default,SUP", XGBClassifier()])
    # class_list.append(["Hist Boost,SUP", HistGradientBoostingClassifier()])
    # class_list.append(["RandomForest 100,SUP", RandomForestClassifier(n_estimators=100, random_state=0)])
    # class_list.append(["Decision Tree,SUP", DecisionTreeClassifier()])
    #
    # # Unsupervised
    # class_list.append(["COPOD,UNS", COPOD(contamination=outliers_fraction)])
    # class_list.append(["KMeans,UNS", KMeans(n_clusters=2)])
    # class_list.append(["iForest,UNS", IsolationForest(contamination=outliers_fraction, warm_start=True)])
    # class_list.append(["HBOS,UNS", HBOS(contamination=outliers_fraction)])
    # class_list.append(["MCD,UNS", MCD(contamination=outliers_fraction)])
    # class_list.append(["PCA,UNS", PCA(contamination=outliers_fraction, weighted=True)])
    #
    # # Unsupervised Bagging
    # class_list.append(["B_COPOD,UNS", BaggingClassifier(base_estimator=PYODEstimator(estimator=COPOD(contamination=outliers_fraction)), n_estimators=10)])
    # class_list.append(["B_KMeans,UNS", BaggingClassifier(base_estimator=PYODEstimator(estimator=KMeans(n_clusters=2)), n_estimators=10)])
    # class_list.append(["B_iForest,UNS", BaggingClassifier(base_estimator=PYODEstimator(estimator=IsolationForest(contamination=outliers_fraction, warm_start=True)), n_estimators=10)])
    # class_list.append(["B_HBOS,UNS", BaggingClassifier(base_estimator=PYODEstimator(estimator=HBOS(contamination=outliers_fraction)), n_estimators=10)])
    # class_list.append(["B_MCD,UNS", BaggingClassifier(base_estimator=PYODEstimator(estimator=MCD(contamination=outliers_fraction)), n_estimators=10)])
    # class_list.append(["B_PCA,UNS", BaggingClassifier(base_estimator=PYODEstimator(estimator=PCA(contamination=outliers_fraction, weighted=True)), n_estimators=10)])


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

    print("Accuracy/MCC = " + '{0:.4f}'.format(accuracy) + "/" + '{0:.4f}'.format(mcc) +
          ", rec-unk = " + '{0:.4f}'.format(rec_unk) + " with " + str(len(unk_y)) +
          " (" + '{0:.4f}'.format(100*len(unk_y)/len(te_y)) + "%) unknowns [" + classifier_name + "] time " + str(elapsed_train) + " ms")

    return [classifier_name, elapsed_train, elapsed_test, tp, tn, fp, fn, accuracy, rec, mcc, tp_u, fn_u, rec_unk]


if __name__ == '__main__':

    # Setup Output File
    with open(SCORES_FILENAME, "w") as myfile:
        # Print Header
        myfile.write("datasetName,attLabel,classifierName,classifierType,trainTime,testTime,tp,tn,fp,fn,acc,rec,mcc,tp_u,fn_u,rec_unk\n")

    for csv_file in glob.glob(DATASETS_DIR + "/*.csv"):

        # Loading tabular dataset
        df, labels, feature_list, att_rate = load_tabular_dataset(csv_file, LABEL_NAME)
        if "/" in csv_file:
            csv_file = csv_file.split("/")[-1]
        elif "\\" in csv_file:
            csv_file = csv_file.split("\\")[-1]

        # Partitioning Train/Test split
        x_tr, x_te, y_tr, y_te = train_test_split(df.drop(columns=[LABEL_NAME]), df[LABEL_NAME],
                                                  test_size=(1-TRAIN_VALIDATION_SPLIT))
        train = x_tr.copy()
        train[LABEL_NAME] = y_tr
        test = x_te.copy()
        test[LABEL_NAME] = y_te

        for label in labels:

            print("\n[" + csv_file + "] Processing label '" + label + "'")

            # Setting up Train/Test set
            if label == NORMAL_TAG:
                tr_x = train.drop(columns=[LABEL_NAME])
                tr_y = np.where(train[LABEL_NAME] == NORMAL_TAG, 0, 1)
                te_x = test.drop(columns=[LABEL_NAME])
                te_y = np.where(test[LABEL_NAME] == NORMAL_TAG, 0, 1)
                unk_x = []
                unk_y = []

            else:
                train_red = train.loc[train[LABEL_NAME] != label]
                tr_x = train_red.drop(columns=[LABEL_NAME])
                tr_y = np.where(train_red[LABEL_NAME] == NORMAL_TAG, 0, 1)
                test_unk = test.loc[test[LABEL_NAME] == label]
                te_x = test.drop(columns=[LABEL_NAME])
                te_y = np.where(test[LABEL_NAME] == NORMAL_TAG, 0, 1)
                unk_x = test_unk.drop(columns=[LABEL_NAME])
                unk_y = np.where(test_unk[LABEL_NAME] == NORMAL_TAG, 0, 1)

            for classifier in available_classifiers(att_rate):

                an_result = data_analysis(classifier, tr_x, tr_y, te_x, te_y, unk_x, unk_y)

                # Write file
                to_print = csv_file + "," + label + "," + ",".join([str(x) for x in an_result])
                with open(SCORES_FILENAME, "a") as myfile:
                    myfile.write(to_print + "\n")

