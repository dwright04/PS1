import optparse, pickle
import numpy as np
import scipy.io as sio
from train_SVM import train_SVM
#from analyse_SVM import measure_FoM
from sklearn import svm
#from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn import preprocessing

def main():

    parser = optparse.OptionParser("[!] usage: python cross_validate_SVM.py -F <data file>")

    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to analyse")

    (options, args) = parser.parse_args()
    dataFile = options.dataFile

    if dataFile == None:
        print parser.usage
        exit(0)

    data = sio.loadmat(dataFile)

    X = data["X"]
    m,n = np.shape(X)
    y = np.squeeze(data["y"])

    kernel_grid = ["rbf", "linear"]
    C_grid = [300,100,30,10,3,1,0.3,0.1]
    gamma_grid = [0.1,0.01]

    #kf = StratifiedKFold(y, n_folds=10, indices=False)
    kf = KFold(m, n_folds=5, indices=False)
    fold = 1
    for kernel in kernel_grid:
        for C in C_grid:
            for gamma in gamma_grid:
                fold=1
                FoMs = []
                for train, test in kf:
                    train_x, train_y = X[train], y[train]
                    scaler = preprocessing.StandardScaler().fit(train_x)
                    train_x = scaler.transform(train_x)
                    print "[*]", fold, kernel, C, gamma
                    file = "cv/SVM_kernel"+str(kernel)+"_C"+str(C)+\
                           "_gamma"+str(gamma)+"_"+dataFile.split("/")[-1].split(".")[0]+\
                           "_fold"+str(fold)+".pkl"
                    try:
                        svm = pickle.load(open(file,"rb"))
                    except IOError:
                        svm = train_SVM(train_x, train_y, kernel, C, gamma)
                        outputFile = open(file, "wb")
                        pickle.dump(svm, outputFile)
                    #FoM, threshold = measure_FoM(X[test], y[test], svm, False)
                    FoM = f1_score(y[test], svm.predict(scaler.transform(X[test])))
                    fold+=1
                    FoMs.append(FoM)
                print "[+] mean FoM: %.3lf" % (np.mean(np.array(FoMs)))
                print

if __name__ == "__main__":
    main()
