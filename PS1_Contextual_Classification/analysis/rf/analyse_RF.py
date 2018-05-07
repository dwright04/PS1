import optparse
import pickle, sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, accuracy_score
import matplotlib as mpl

def measure_FoM(X, y, classifier, plot=True):
    pred = classifier.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, pred)

    FoM = 1-tpr[np.where(fpr<=0.01)[0][-1]]
    print "[+] FoM: %.4f" % (FoM)
    threshold = thresholds[np.where(fpr<=0.01)[0][-1]]
    print "[+] threshold: %.4f" % (threshold)
    print

    if plot:
        font = {"size": 18}
        plt.rc("font", **font)
        plt.rc("legend", fontsize=14)
    
        plt.xlabel("Missed Detection Rate (MDR)")
        plt.ylabel("False Positive Rate (FPR)")
        plt.yticks([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
        plt.ylim((0,1.05))

        plt.plot(1-tpr, fpr, "k-", lw=5)
        plt.plot(1-tpr, fpr, color="#FF0066", lw=4)

        plt.plot([x for x in np.arange(0,FoM+1e-3,1e-3)], \
                  0.01*np.ones(np.shape(np.array([x for x in np.arange(0,FoM+1e-3,1e-3)]))), \
                 "k--", lw=3)

        plt.plot(FoM*np.ones((11,)), [x for x in np.arange(0,0.01+1e-3, 1e-3)], "k--", lw=3)

        plt.xticks([0, 0.05, 0.10, 0.25, FoM], rotation=70)

        locs, labels = plt.xticks()
        plt.xticks(locs, map(lambda x: "%.3f" % x, locs))
        plt.show()
    return FoM, threshold

def main():

    parser = optparse.OptionParser("[!] usage: python analyse_RF.py -F <data file>"+\
                                   " -c <classifier file> -s <data set>")

    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to analyse")
    parser.add_option("-c", dest="classifierFile", type="string", \
                      help="specify classifier to use")
    parser.add_option("-s", dest="dataSet", type="string", \
                      help="specify data set to analyse ([training] or [test] set)")

    (options, args) = parser.parse_args()
    dataFile = options.dataFile
    classifierFile = options.classifierFile
    dataSet = options.dataSet

    print

    if dataFile == None or classifierFile == None or dataSet == None:
        print parser.usage
        exit(0)

    if dataSet != "training" and dataSet != "test":
        print "[!] Exiting: data set must be 1 of 'training' or 'test'"
        exit(0)

    try:
        data = sio.loadmat(dataFile)
    except IOError:
        print "[!] Exiting: %s Not Found" % (dataFile)
        exit(0)

    if dataSet == "training":
        X = data["X"]
        y = np.squeeze(data["y"])
    elif dataSet == "test":
        X = data["testX"]
        y = np.squeeze(data["testy"])

    try:
        classifier = pickle.load(open(classifierFile, "rb"))
    except IOError:
        print "[!] Exiting: %s Not Found" % (classifierFile)
        exit(0)
    #measure_FoM(X, y, classifier)
    pred = classifier.predict(X)
    print pred
    output = open("rf_preds.txt","w")
    print classifier.predict_proba(X)[:,1]
    for p in classifier.predict_proba(X)[:,1]:
        output.write("%.3f\n" % p)
    output.close()
    print "f1 score : %.3f" % f1_score(y, pred)
    print "acc. score : %.3f" % accuracy_score(y, pred)

    #print X[np.where(y != pred)[0],7]
    print np.where(y != pred)[0]

    print y[np.where(y != pred)]

    pos= pred[y==1]
    neg = pred[y==0]
 
 
    tps = np.where(pos==1)[0]
    fns = np.where(pos!=1)[0]

    tns = np.where(neg==0)[0]
    fps = np.where(neg!=0)[0]
    
    font = {"size"   : 26}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=22)
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.scatter(X[y==1,:][tps,7],\
             X[y==1,:][tps,4],\
             color="#3A86FF", edgecolor="none",alpha=.5, s=100, label="true positive")

    plt.scatter(X[y==0,:][tns,7],\
             X[y==0,:][tns,4],\
             color="#FF006E", edgecolor="none",alpha=.5, s=100, label="true negative")
    plt.scatter(X[y==0,:][fps,7],\
             X[y==0,:][fps,4],color="#FFBE0B", s=100, edgecolor="none",label="false positive")

    plt.scatter(X[y==1,:][fns,7],\
         X[y==1,:][fns,4],color="#89FC00", s=100, edgecolor="none",label="false negative")

    plt.plot([0.5,0.5],[-1,10],"k--")


    plt.tick_params(axis='both', which='major', labelsize=18)
    mpl.rcParams['legend.scatterpoints'] = 1
    plt.legend(loc="lower center", numpoints=1)
    #plt.axes().set_aspect('equal', 'datalim')
    
    # Hide the right and top spines
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
    plt.axes().yaxis.set_ticks_position('left')
    plt.axes().xaxis.set_ticks_position('bottom')
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.xlabel("offset [arcsec]")
    plt.ylabel("photo. redshift")
    plt.xticks([0,0.5,1,2,3,4,5])
    plt.xlim(xmin=-.1)
    plt.ylim(ymin=-.03,ymax=0.61)
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()
