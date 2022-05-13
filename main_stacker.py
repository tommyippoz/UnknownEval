import copy

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
from utils.Stacking import Stacker
from utils.dataset_utils import load_tabular_dataset
from utils.main_utils import fast_unsupervised_classifiers, data_analysis

DATASETS_DIR = "datasets"
SCORES_FILENAME = "output/unk_scores_stacker.csv"

NORMAL_TAG = 'normal'
LABEL_NAME = 'multilabel'

TRAIN_VALIDATION_SPLIT = 0.5


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

            base_learners = fast_unsupervised_classifiers(att_rate)
            for name, learner in base_learners:
                print("[Stacking] Training base-learner '" + name + "'")
                learner.fit(tr_x, tr_y)

            for classifier in fast_unsupervised_classifiers(att_rate):

                an_result, y_pred, y_unk = data_analysis(copy.deepcopy(classifier), tr_x, tr_y, te_x, te_y, unk_x, unk_y)
                to_print = csv_file + "," + label + "," + ",".join([str(x) for x in an_result])
                with open(SCORES_FILENAME, "a") as myfile:
                    myfile.write(to_print + "\n")

                stacking = ["Stacking-" + classifier[0].split(",")[0] + "-noT,Custom",
                            Stacker(base_level_learners=base_learners,
                                    meta_level_learner=copy.deepcopy(classifier[1]), use_training=False)]
                an_result, y_pred, y_unk = data_analysis(stacking, tr_x, tr_y, te_x, te_y, unk_x, unk_y)
                to_print = csv_file + "," + label + "," + ",".join([str(x) for x in an_result])
                with open(SCORES_FILENAME, "a") as myfile:
                    myfile.write(to_print + "\n")

                stacking = ["Stacking-" + classifier[0].split(",")[0] + ",Custom",
                            Stacker(base_level_learners=base_learners,
                                    meta_level_learner=copy.deepcopy(classifier[1]), use_training=True)]
                an_result, y_pred, y_unk = data_analysis(stacking, tr_x, tr_y, te_x, te_y, unk_x, unk_y)
                to_print = csv_file + "," + label + "," + ",".join([str(x) for x in an_result])
                with open(SCORES_FILENAME, "a") as myfile:
                    myfile.write(to_print + "\n")

                print("")
                classifier = None
