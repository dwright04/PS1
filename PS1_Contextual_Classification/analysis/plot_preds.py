import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx

nn = []
offsets = []
for line in open("nn/nn_preds.txt", "r").readlines():
    data = line.rstrip().split(",")
    nn.append(float(data[0]))
    offsets.append(float(data[1]))

rf = []

for line in open("rf/rf_preds.txt", "r").readlines():
    rf.append(float(line.rstrip()))

print nn
print rf

nn = np.array(nn)
rf = np.array(rf)
offsets = np.array(offsets)
data = sio.loadmat("contextual_classification_dataset_20150706_shuffled.mat")
y = np.squeeze(data["y"])

zipped = zip(offsets, nn, rf, y)
zipped.sort(key = lambda x : x[0])

font = {"size"   : 26}
plt.rc("font", **font)
plt.rc("legend", fontsize=22)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
cmap="cool_r"
cNorm = colors.Normalize(vmin=np.min(offsets), vmax=np.max(offsets))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
i=0
for element in zipped:
    colorVal = scalarMap.to_rgba(element[0])
    if element[3] == 1:
        plt.scatter(element[1], element[2], color=colorVal, s=100, marker="o", alpha=.3)
    elif element[3] == 0:
        plt.scatter(element[1], element[2], color=colorVal, s=100, marker="^", alpha=.3)
plt.scatter(-10, -10, color=colorVal, s=100, marker="o", alpha=1,label="SNe")
plt.scatter(-10, -10, color=colorVal, s=100, marker="^", alpha=1,label="not SNe")
scalarMap.set_array(offsets)
cbar = plt.colorbar(scalarMap)
cbar.set_label("offset [arcsec]",size=22)
cbar.ax.tick_params(labelsize=18)
#plt.scatter(nn[y==1],rf[y==1],color="#3A86FF", s=50, label="SNe")
#plt.scatter(nn[y==0],rf[y==0],color="#FF006E", s=50, label="not SNe")
plt.tick_params(axis='both', which='major', labelsize=18)
mpl.rcParams['legend.scatterpoints'] = 1
plt.legend(loc="lower right", numpoints=1)
#plt.axes().set_aspect('equal', 'datalim')
# Hide the right and top spines
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
plt.axes().yaxis.set_ticks_position('left')
plt.axes().xaxis.set_ticks_position('bottom')
plt.tick_params(axis='both', which='major', labelsize=22)
plt.xlabel("ANN hypotheses")
plt.ylabel("RF hypotheses")
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()