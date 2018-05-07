import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def main():
    savedRf = "RF_n_estimators100_max_features4_min_samples_leaf1_contextual_classification_dataset_20150706_shuffled.pkl"

    rf = pickle.load(open(savedRf, "rb"))
    print np.sum(rf.feature_importances_)
    print rf.feature_importances_
    rf.feature_importances_
    image = np.ones((1,len(rf.feature_importances_)))
    image[0,:] *= rf.feature_importances_
    #image[1,:] *= rf.feature_importances_
    print np.shape(image)

    font = {"size"   : 26}
    plt.rc("font", **font)
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    im = plt.imshow(image, interpolation="nearest", cmap="PuBuGn_r")
    plt.axis("off")
    
    plt.text(-0.1,0.1,"u-g",color="w")
    plt.text(0.9,0.1,"g-r",color="w")
    plt.text(1.95,0.1,"r-i",color="w")
    plt.text(2.9,0.1,"i-z",color="w")
    plt.text(3.6,0.16,"  photo.\nredshift",color="w")
    plt.text(4.7,0.1,"galaxy?",color="w")
    plt.text(5.8,0.1,"qso?",color="w")
    plt.text(6.7,0.16,"  offset\n[arcsec]",color="k")
    plt.text(7.7,0.1,r"$\Delta$mag.",color="w")
    
    cbar = plt.colorbar(im, orientation="horizontal",pad=0.05)
    cbar.set_label("relative importance")
    
    plt.show()

if __name__ == "__main__":
    main()
