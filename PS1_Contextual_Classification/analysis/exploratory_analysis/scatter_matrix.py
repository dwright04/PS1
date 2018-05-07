import itertools
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rcParams

def scatterplot_matrix(data, names, flag=0, fig=None, **kwargs):

    mask = np.tril_indices(9)
    print np.shape(mask)
    font = {"size"   : 16}
    plt.rc("font", **font)
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    print np.shape(data)
    numvars, numdata = data.shape
    #print
    #print fig == None
    #print
    if fig == None:
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(9,9))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        axs = axes.flat
        new=True
    else:
        axes = np.reshape(fig.axes, (numvars,numvars))
        axs = fig.axes
        new = False
    if new:
        for ax in axs:
            # Hide all ticks and labels
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    counter = 0
    for i in range(numvars):
        #print not new
        if not new and flag == 1:
            #axes[i,i].text(0,0,names[i], ha="center", va="center")
            axes[i,i].set_xlim(np.min(data[i,:])-.1*np.max(data[i,:]),np.max(data[i,:])+.1*np.max(data[i,:]))
            axes[i,i].set_ylim(np.min(data[i,:])-.1*np.max(data[i,:]),np.max(data[i,:])+.1*np.max(data[i,:]))
            rcParams['xtick.direction'] = 'in'
            rcParams['ytick.direction'] = 'in'
            axes[i,i].xaxis.set_visible(True)
            axes[i,i].yaxis.set_visible(True)
            axes[i,i].text(np.mean([np.min(data[i,:])-.1*np.max(data[i,:]),np.max(data[i,:])+.1*np.max(data[i,:])]),\
                           np.mean([np.min(data[i,:])-.1*np.max(data[i,:]),np.max(data[i,:])+.1*np.max(data[i,:])]),\
                           names[i], ha="center", va="center")
            axes[i,i].axis("off")
            #print axes[i,i].axis.invert_ticklabel_direction()
            #axes[i,i].invert_ticklabel_direction()
            #for tick in axes[i,i].yaxis.get_major_ticks():
            #    tick.set_pad(-20)
            #for tick in axes[i,i].xaxis.get_major_ticks():
            #    tick.set_pad(-20)
            #axes[i,i].set_xticks(axes[i,i].get_xticks()[1:-1])

        for j in range(numvars):
            if i==j:
                counter +=1
                continue
            print flag >= 1 and i == mask[0][counter] and j == mask[1][counter]
            if flag >= 1 and i == mask[0][counter] and j == mask[1][counter]:
                axes[i,j].scatter(data[j,:],data[i,:],**kwargs)
                axes[i,j].set_axis_bgcolor('#FFDF64')
                axes[i,j].patch.set_alpha(0.5)
                if i == 7:
                    xmin, xmax = axes[i,j].get_xlim()
                    axes[i,j].plot([xmin, xmax],[.5,.5],"k--")
                    axes[i,j].set_xlim(xmin,xmax)
                if i==8 and j == 7:
                    ymin, ymax = axes[i,j].get_ylim()
                    axes[i,j].plot([.5,.5],[ymin, ymax],"k--")
                    axes[i,j].set_ylim(ymin,ymax)
                counter += 1
            
            elif flag == 0:
                axes[i,j].scatter(data[j,:],data[i,:],**kwargs)

    return fig

def scatterplot_matrix_classes(data, y, names, pred=None, **kwargs):
    #print np.where(y==0)[1]
    #X = data[np.where(y==1)[1],:].T
    X = data[np.where(y==1)[1],:].T
    fig = scatterplot_matrix(X, names, 0, None, color="#3A86FF",\
                             marker="o", s=3,alpha=.75)

    X = data[np.where(y==0)[1],:].T
    #print fig
    fig = scatterplot_matrix(X, names, 0, fig, color="#FF006E",\
                             marker="o", edgecolor="none",s=3,alpha=.75)

    data = sio.loadmat("../contextual_classification_dataset_20150706_shuffled.mat")
    print data.keys()
    data = data["testX"]
    #print X[:,5]
    #y = data["testy"]
    NN_misses = np.array([32, 64, 76, 118, 124])
    NN_labels = np.array([1, 0, 1, 1, 1])
    RF_misses = np.array([31, 32, 56, 65, 124])
    RF_labels = np.array([1, 1, 0, 0, 1])
    print NN_misses[NN_labels==0]
    #false_positives = np.concatenate((NN_misses[NN_labels==0],RF_misses[RF_labels==0]))
    print NN_misses[NN_labels==1]
    #false_negatives = np.concatenate((NN_misses[NN_labels==1],RF_misses[RF_labels==1]))

    #X = data[false_positives,:].T
#    fig = scatterplot_matrix(X, names, 0, fig, color="none",\
#                         marker="o", edgecolor="k",s=10,alpha=.75)
#    X = data[false_negatives,:].T
    X = data[NN_misses[NN_labels==0],:].T
    fig = scatterplot_matrix(X, names, 2, fig, color="#89FC00",\
                             marker="o", edgecolor="k",s=35,alpha=1)
    X = data[NN_misses[NN_labels==1],:].T
    fig = scatterplot_matrix(X, names, 2, fig, color="#FFBE0B",\
                             marker="o", edgecolor="k",s=35,alpha=1)

    X = data[RF_misses[RF_labels==0],:].T
    fig = scatterplot_matrix(X, names, 2, fig, color="#89FC00",\
                             marker="^", edgecolor="k",s=35,alpha=1)
    X = data[RF_misses[RF_labels==1],:].T
    fig = scatterplot_matrix(X, names, 1, fig, color="#FFBE0B",\
                             marker="^", edgecolor="k",s=35,alpha=1)
    fig.set_facecolor("white")

def main():
    #data = sio.loadmat("../contextual_classification_photo_dataset.mat")
    #X = data["X"]
    #print X[:,5]
    #y = data["y"]
    
    data = sio.loadmat("../contextual_classification_dataset_20150706_shuffled.mat")
    X = data["X"]
    #print X[:,5]
    y = data["y"]
    
    #X = np.concatenate((X,X1))
    #print np.shape(y)
    #print np.shape(y1)
    #y = np.concatenate((np.squeeze(y),np.squeeze(np.ones(np.shape(y1)))))[np.newaxis]
    names = ["u-g","g-r","r-i","i-z","photo.\nredshift","galaxy?",\
             "qso?", "offset\n[arcsec]", r"$\Delta$mag."]
    scatterplot_matrix_classes(X,y,names)

    plt.show()

if __name__ == "__main__":
    main()