import utils, model, sklearn
import numpy as np

scores = []

for i in range(10):
    svm = model.Model_SVM()
    svm.split_train_test()
    svm.train()
    score = svm.get_score()
    scores.append(score)
    print(i, score)

print(sum(scores)/len(scores))