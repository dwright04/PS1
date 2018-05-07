import optparse, pickle
import numpy as np
import scipy.io as sio
from train_RF import train_RF
#from analyse_RF import measure_FoM
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.metrics import f1_score

def main():

    parser = optparse.OptionParser("[!] usage: python cross_validate_RF.py -F <data file>")

    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to analyse")

    (options, args) = parser.parse_args()
    dataFile = options.dataFile

    if dataFile == None:
        print parser.usage
        exit(0)

    data = sio.loadmat(dataFile)
    #X = np.concatenate((data["X"], data["validX"]))
    X = data["X"]
    m,n = np.shape(X)
    y = np.squeeze(data["y"])
    #y = np.squeeze(np.concatenate((data["y"], data["validy"])))
    n_estimators_grid = [10, 100, 1000]
    max_features_grid = [i for i in range(1,n+1)]
    min_samples_leaf_grid = [1, 2, 5]

    #kf = StratifiedKFold(y, n_folds=10, indices=False)
    kf = KFold(m, n_folds=5, indices=False)
    fold = 1
    for n_estimators in n_estimators_grid:
        for max_features in max_features_grid:
            for min_samples_leaf in min_samples_leaf_grid:
                fold=1
                FoMs = []
                for train, test in kf:
                    print "[*]", fold, n_estimators, max_features, min_samples_leaf
                    file = "cv/RF_n_estimators"+str(n_estimators)+"_max_features"+str(max_features)+\
                           "_min_samples_leaf"+str(min_samples_leaf)+"_"+dataFile.split("/")[-1].split(".")[0]+\
                           "_fold"+str(fold)+".pkl"
                    try:
                        rf = pickle.load(open(file,"rb"))
                    except IOError:
                        train_x, train_y = X[train], y[train]
                        rf = train_RF(train_x, train_y, n_estimators, max_features, min_samples_leaf)
                        outputFile = open(file, "wb")
                        pickle.dump(rf, outputFile)
                    #FoM, threshold = measure_FoM(X[test], y[test], rf, False)
                    FoM = f1_score(y[test], rf.predict(X[test]))
                    fold+=1
                    FoMs.append(FoM)
                print "[+] mean FoM: %.3lf" % (np.mean(np.array(FoMs)))
                print

if __name__ == "__main__":
    main()
