import os, librosa, h5py, math
import numpy as np

'''
For low-level acoustic features, we extract 32 features for
every frame: F0 (pitch), voice probability, zero-crossing rate,
12-dimensional Mel-frequency cepstral coefficients (MFCC)
with log energy, and their first time derivatives. In the DNNbased framework, we used as a baseline, those 32-dimensional
vectors are expanded to 800-dimensional vectors using the context window with the size of 250 ms. The network contains
3 hidden layers and each hidden layer has 256 nodes, and
the weights were trained by back-propagation algorithm using
stochastic gradient descent with mini-batch of 128 samples. In
the RNN-based system, the 32-dimensional vectors are directly
used for input. The network contains 2 hidden layers with 128
BLSTM cells (64 forward nodes and 64 backward nodes). Later
experiments showed that the performance did not improve with
higher number of hidden layers and nodes in both DNN-based
and RNN-based systems. The reason is most probably overfitting caused by data insufficiency.
'''

SR = 48000
HOP = 0.20
HOP_LENGTH = int(HOP*SR)
LABELS = {1:"neutral", 2:"calm", 3:"happy", 4:"sad", 5:"angry", 6:"fearful", 7:"disgust", 8:"surprised"}

def get_dataset_stats():
    length = []
    labels = [0]*8
    for actor_idx in range(24):
        for file_idx, file_name in enumerate(list(filter(lambda x: "wav" in x, os.listdir(f"datasets/RAVDESS/Actor_{(actor_idx+1):02d}")))):
            print(f"\rProcessing Actor {(actor_idx+1):02d}/24, File {(file_idx+1):02d}/60", end="")
            labels[int(file_name[:-4].split("-")[2])-1] += 1
            audio = librosa.load(f"datasets/RAVDESS/Actor_{(actor_idx+1):02d}/{file_name}", sr=SR)[0]
            length.append(len(librosa.effects.trim(audio)[0]))
    mean_length = sum(length)//len(length)
    min_length = min(length)
    max_length = max(length)
    assert sum(labels) == len(length)
    print("\rDataset Statistics:"+" "*15)
    print(f"  Sample Length: mean={mean_length}, min={min_length}, max={max_length}")
    print(f"  Labels: total={sum(labels)}, "+", ".join([f"{LABELS[i+1]}={labels[i]}" for i in range(len(labels))]))
    return [mean_length, min_length, max_length], labels


def get_stats_tess():
    import time
    path = ['datasets/TESS', 'datasets/RAVDESS']
    lst = []
    start_time = time.time()

    for p in path:
        for subdir, dirs, files in os.walk(p):
            for file in files:
                try:
                    #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                    print(os.path.join(subdir, file))
                    X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(
                        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                    # print(np.shape(mfccs))
                    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                    file = int(subdir[-1:]) - 1
                    arr = mfccs, file
                    lst.append(arr)
                # If the file is not valid, skip it
                except ValueError:
                    continue
    print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
    print(np.shape(lst))
    X, y = zip(*lst)
    print(np.shape(X), np.shape(y))
    with h5py.File('tess_features', 'w') as tess:
        tess.create_dataset(name='features', data=X)
        tess.create_dataset(name='labels', data=y)
    return np.asarray(X), np.asarray(y)

def get_feature_vector(file_name, length, n_mfcc, flatten=False):
    audio = librosa.effects.trim(librosa.load(f"datasets/RAVDESS/{file_name}", sr=SR)[0])[0]
    padding = length-len(audio)
    audio = np.concatenate((np.zeros(padding//2), audio, np.zeros(padding-padding//2)))
    assert len(audio) == length
    feature_vector = np.empty((n_mfcc, math.ceil(length/HOP_LENGTH)))
    feature_vector[:n_mfcc] = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=n_mfcc, hop_length=HOP_LENGTH)
    '''
    todo: F0, Intensity, Power, etc.
    '''
    if flatten:
        return feature_vector.T.flatten()
    return feature_vector

def extract_features(n_mfcc, flatten=False):
    with h5py.File("dataset_stats.h5", "r") as stats:
        length = stats['sample_length'][2]
    num_samples = 24*60
    labels = np.empty(num_samples)
    labels_onehot = np.zeros((num_samples, 8))
    if flatten:
        features = np.empty((num_samples, n_mfcc*26))
    else:
        features = np.empty((num_samples, n_mfcc, 26))
    for actor_idx in range(24):
        for file_idx, file_name in enumerate(list(filter(lambda x: "wav" in x, os.listdir(f"datasets/RAVDESS/Actor_{(actor_idx+1):02d}")))):
            print(f"\rProcessing Actor {(actor_idx+1):02d}/24, File {(file_idx+1):02d}/60", end="")
            labels[actor_idx*60+file_idx] = int(file_name[:-4].split("-")[2])
            labels_onehot[actor_idx*60+file_idx, int(file_name[:-4].split("-")[2])-1] = 1
            features[actor_idx*60+file_idx] = get_feature_vector(f"Actor_{(actor_idx+1):02d}/{file_name}", length, n_mfcc, flatten)
    print("\nFinish Extracting Features")
    return features, labels, labels_onehot

def init(normalize=True):
    if "dataset_stats.h5" not in os.listdir():
        print("Getting Dataset Statistics")
        length, labels = get_dataset_stats()
        with h5py.File("dataset_stats.h5", "w") as f:
            f.create_dataset(name="sample_length", data=length)
            f.create_dataset(name="labels", data=labels)
        print("Dataset Statistics Stored to /dataset_stats.h5")
    fname = "features_norm.h5" if (normalize) else "features.h5"
    if fname not in os.listdir():
        print("Extracting Features")
        features, labels, labels_onehot = extract_features(13, False) # unflat false
        if (normalize): 
            # take means INDIVIDUALLY - better for local mfcc
            # https://www.kaggle.com/c/freesound-audio-tagging/discussion/54082
            # rshape = np.reshape(features, (features.shape[0], -1))
            # print("finished reshaping")
            # print(np.shape(features))
            mean = np.mean(features, axis=(1,2), keepdims=True) # (n,1,1,1)
            # print(mean.shape)
            std = np.std(features, axis=(1,2), keepdims=True)   # (n,1,1,1)
            # print(std.shape)
            # mean =  mean[:, np.newaxis, np.newaxis, np.newaxis]
            # std = std[:, np.newaxis, np.newaxis, np.newaxis]
            print(np.shape(mean), print(np.shape(std)))
            # return
            features = (features - mean) / std 
            print("normalized features... ", np.shape(features))
        with h5py.File(fname, "w") as f:
            f.create_dataset(name="features", data=features)
            f.create_dataset(name="labels", data=labels)
            f.create_dataset(name="labels_onehot", data=labels_onehot)
        print("Features Extracted")
        
if __name__ == '__main__':
    # init(normalize=False)
    #init(normalize=True)
    # get_dataset_stats()
    get_stats_tess()
