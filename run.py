import utils, model, sklearn, h5py
import numpy as np
import matplotlib.pyplot as plt

scores = []
scores_mfcc = []

utils.init()

# accuracy based on n_mfcc
def n_mfcc():
    f = h5py.File("outputs/mfcc_svm.h5", "w")
    for n_mfcc in range(1, 50):
        print(n_mfcc)
        features, labels = utils.extract_features(n_mfcc, flatten=True)[:2]
        for i in range(10):
            print(n_mfcc, i)
            svm = model.Model_SVM(labels, features, 0.33)
            svm.split_train_test()
            svm.train()
            score = svm.get_score()
            scores.append(score)
        print(sum(scores)/len(scores))
        scores_mfcc.append(sum(scores)/len(scores))
    f.create_dataset(name="accuracy/n_mfcc", data=scores_mfcc)
    f.close()

def plot_n_mfcc():
    with h5py.File("outputs/mfcc_svm.h5", "r") as f:
        y = f["accuracy/n_mfcc"][()]
        x = np.arange(1, len(y)+1)
    plt.plot(x, y)
    plt.xlabel("n_mfcc")
    plt.ylabel("accuracy")
    plt.show()

def cnn():
    f = h5py.File("outputs/mfcc_svm.h5", "w")
    n_mfcc = 13 # change this
    print(f)
    # features, labels = utils.extract_features(n_mfcc, flatten=False)[:2]
    with h5py.File("features_unflat.h5", "r") as f:
        print(f.keys())
        features = np.array(f["features"])
        labels = np.array(f["labels_onehot"])
    print(np.shape(features), np.shape(labels))
    cnn = model.cnn(labels, features, 0.33)
    cnn.split_train_test()
    cnn.build_model()
    cnn.train()
    cnn.evaluate()


if __name__ == '__main__':
    # n_mfcc()
    #plot_n_mfcc()
    cnn()
