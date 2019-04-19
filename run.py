import utils, model, sklearn
import numpy as np

labels, features = utils.extract_labels_and_features()

for i in range(1):
    # cross validation train 0.8 test 0.2
    labels, features = utils.shuffle_labels_and_features(labels, features)
    train_x, validate_x = np.split(features, int(len(features)*0.8), int(len(features)*0.2))
    train_y, validate_y = np.split(labels, int(len(labels)*0.8), int(len(labels)*0.2))
    print(train_x, validate_x, train_y, validate_y)