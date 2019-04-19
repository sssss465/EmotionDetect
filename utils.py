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

def get_feature_vector(file_name, length):
    audio = librosa.effects.trim(librosa.load(f"datasets/RAVDESS/{file_name}", sr=SR)[0])[0]
    padding = length-len(audio)
    audio = np.concatenate((np.zeros(padding//2), audio, np.zeros(padding-padding//2)))
    assert len(audio) == length
    feature_vector = np.empty((13, math.ceil(length/HOP_LENGTH)))
    feature_vector[:13] = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13, hop_length=HOP_LENGTH)
    '''
    todo: F0, Intensity, Power, etc.
    '''
    return feature_vector

def extract_features():
    with h5py.File("dataset_stats.h5", "r") as stats:
        length = stats['sample_length'][2]
    num_samples = 24*60
    labels = np.empty(num_samples)
    labels_onehot = np.zeros((num_samples, 8))
    features = np.empty((num_samples, 13, 26))
    for actor_idx in range(24):
        for file_idx, file_name in enumerate(list(filter(lambda x: "wav" in x, os.listdir(f"datasets/RAVDESS/Actor_{(actor_idx+1):02d}")))):
            print(f"\rProcessing Actor {(actor_idx+1):02d}/24, File {(file_idx+1):02d}/60", end="")
            labels[actor_idx*60+file_idx] = int(file_name[:-4].split("-")[2])
            labels_onehot[actor_idx*60+file_idx, int(file_name[:-4].split("-")[2])-1] = 1
            features[actor_idx*60+file_idx] = get_feature_vector(f"Actor_{(actor_idx+1):02d}/{file_name}", length)
    print(labels_onehot[0])
    print("Finish Extracting Features")
    return labels, features

def shuffle_labels_and_features(labels, features):
    assert len(labels) == len(features)
    rand = np.arange(len(labels))
    np.random.shuffle(rand)
    return labels[rand], features[rand]

def init():
    if "dataset_stats.h5" not in os.listdir():
        print("Getting Dataset Statistics")
        length, labels = get_dataset_stats()
        f = h5py.File("dataset_stats.h5", "w")
        f.create_dataset(name="sample_length", data=length, dtype="int32")
        f.create_dataset(name="labels", data=labels, dtype="uint8")
        print("Dataset Statistics Stored to /dataset_stats.h5")
    if "features.h5" not in os.listdir():
        extract_features()

if __name__ == '__main__':
    init()