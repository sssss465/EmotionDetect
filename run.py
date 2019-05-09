import utils, model, sklearn, h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

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

def cnn(n_mfcc = 13):
    f = h5py.File("outputs/mfcc_svm.h5", "w")
    print(f)
    if (n_mfcc == 13):
        with h5py.File("features_norm.h5", "r") as f:
            print(f.keys())
            features = np.array(f["features"])
            labels = np.array(f["labels_onehot"])
    else:
        # features, _, labels = utils.extract_features(n_mfcc, flatten=False)
        # features = np.mean(features.T, axis=0).T
        # with h5py.File('ravdess_40_mfcc', 'w') as f:
        #     f.create_dataset('features', data=features)
        #     f.create_dataset('labels', data=labels)
        with h5py.File('ravdess_40_mfcc', 'r') as f:
            features = f['features'][:]
            labels = f['labels'][:]
        print(np.shape(features), np.shape(labels))
        with h5py.File("tess_features", "r") as f:
            f2 = np.array(f['features'])
            l2 = np.array(f['labels'])
            l2 = keras.utils.to_categorical(l2, num_classes=8)
            print(np.shape(f2), np.shape(l2))
   
    features = np.concatenate((features, f2), axis=0)
    labels = np.concatenate((labels, l2), axis=0)
    features = np.concatenate((features, np.zeros((features.shape[0],400))), axis=1)
    features = np.reshape(features, (features.shape[0], 40, -1))
    # feature augmentation
    features = np.concatenate((features, np.flip(features, axis=1)),axis=0)
    labels = np.concatenate((labels, labels), axis=0)
    print(np.shape(features), np.shape(labels))
    cnn = model.cnn(labels, features, 0.20)
    cnn.split_train_test()
    cnn.build_model()
    cnn.train()
    cnn.evaluate()


if __name__ == '__main__':
    # n_mfcc()
    #plot_n_mfcc()
    import sys
    print(sys.version)
    cnn(40)
