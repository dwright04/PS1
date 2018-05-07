import scipy.io as sio
from sklearn.cross_validation import StratifiedKFold

def main():
    data = sio.loadmat("../contextual_classification_dataset_20150521.mat")

    X = data["X"]
    y = data["y"]

    kf = StratifiedKFold(np.squeeze(y), n_folds=3, indices=False)
    fold = 1
    for train, test in kf:
        train_x, train_y = X[train], y[train]
if __name__ == "__main__":
    main()