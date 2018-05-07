\import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import roc_curve

def on_pick(event):
    N = len(event.ind)
    if not N: return True

    print y[event.ind[0]]

def main():
    global ids, surveys, y
    dataPath = "/Users/dew/development/PS1-Class/data/"
    dataFile = "context_data_set_maglt21_offset_host_r_type_20150402.mat"
    data = sio.loadmat(dataPath+dataFile)
    
    X = data["X"]
    X_scaled = preprocessing.scale(X)
    y = data["y"]
    
    ids = [str(x).rstrip() for x in data["ids"]]
    surveys = [str(x).rstrip() for x in data["surveys"]]

    print len(np.where(y==0)[0])
    print len(np.where(y==1)[0])
    print len(np.where(y==2)[0])
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    #print X_r

    #h = .02
    #C = 100
    #y_r = np.copy(y)
    #y_r[y_r==2] -= 2
    #svc = svm.SVC(kernel='poly', C=C).fit(X_r, np.squeeze(y_r))
    # create a mesh to plot in
    #x_min, x_max = X_r[:, 0].min() - h, X_r[:, 0].max() + h
    #y_min, y_max = X_r[:, 1].min() - h, X_r[:, 1].max() + h
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                     np.arange(y_min, y_max, h))

    #Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    #print Z
    # Put the result into a color plot
    #Z = Z.reshape(xx.shape)

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.set_xlabel("sdss host offset [arcsecs]")
    #ax1.set_ylabel("sdss host qso colours [True, False]")
    ax1.set_ylabel("sdss host r mag")
    ax1.scatter(X[np.where(y==0)[0],0], X[np.where(y==0)[0],1], color="#3366FF", label="AGN")
    ax1.scatter(X[np.where(y==1)[0],0], X[np.where(y==1)[0],1], color="#66FF33", label="SN")
    ax1.scatter(X[np.where(y==2)[0],0], X[np.where(y==2)[0],1], color="#FF0066", label="star")

    ax2 = fig.add_subplot(222)
    ax2.set_xlabel("sdss host offset [arcsecs]")
    ax2.set_ylabel("sdss galactic host [True, False]")
    ax2.scatter(X[np.where(y==0)[0],0], X[np.where(y==0)[0],2], color="#3366FF")
    ax2.scatter(X[np.where(y==1)[0],0], X[np.where(y==1)[0],2], color="#66FF33")
    ax2.scatter(X[np.where(y==2)[0],0], X[np.where(y==2)[0],2], color="#FF0066")

    ax3 = fig.add_subplot(223)
    #ax3.set_xlabel("sdss host qso colours [True, False]")
    ax3.set_xlabel("sdss host r mag")
    ax3.set_ylabel("sdss galactic host [True, False]")
    ax3.scatter(X[np.where(y==0)[0],1], X[np.where(y==0)[0],2], color="#3366FF")
    ax3.scatter(X[np.where(y==1)[0],1], X[np.where(y==1)[0],2], color="#66FF33")
    ax3.scatter(X[np.where(y==2)[0],1], X[np.where(y==2)[0],2], color="#FF0066")

    ax4 = fig.add_subplot(224)
    ax4.set_xlabel("pca first component")
    ax4.set_ylabel("pca second component")
    #ax4.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    ax4.scatter(X_r[np.where(y==0)[0],0], X_r[np.where(y==0)[0],1], color="#3366FF", picker=5)
    ax4.scatter(X_r[np.where(y==1)[0],0], X_r[np.where(y==1)[0],1], color="#66FF33", picker=5)
    ax4.scatter(X_r[np.where(y==2)[0],0], X_r[np.where(y==2)[0],1], color="#FF0066", picker=5)
    #print X_r[np.where(y==0)[0],:]
    #ax4.plot(X_r[np.where(y==0)[0],:], np.ones(np.shape(X_r[np.where(y==0)[0],:])), ".", ms=7, color="#3366FF", picker=5)
    #ax4.plot(X_r[np.where(y==1)[0],:], np.ones(np.shape(X_r[np.where(y==1)[0],:])), ".", ms=7, color="#66FF33", picker=5)
    #ax4.plot(X_r[np.where(y==2)[0],:], np.ones(np.shape(X_r[np.where(y==2)[0],:])), ".", ms=7, color="#FF0066", picker=5)

    ax1.legend()
    #fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()

if __name__ == "__main__":
    main()
