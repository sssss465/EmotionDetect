import utils, model, sklearn, h5py
import numpy as np

scores = []
scores_mfcc = []

utils.init()

# accuracy based on n_mfcc

f = h5py.File("mfcc_svm.h5", "w")

for n_mfcc in range(1, 51):
    print(n_mfcc)
    features, labels, e = utils.extract_features(n_mfcc, flatten=True)
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
