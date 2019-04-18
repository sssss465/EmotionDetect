import os, librosa
import numpy as np
import sklearn

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
WAV_LENGTH = 240765
HOP = 0.20
HOP_LENGTH = int(HOP*SR)
LABELS = {1:"neutral", 2:"calm", 3:"happy", 4:"sad", 5:"angry", 6:"fearful", 7:"disgust", 8:"surprised"}

def get_dataset_stats():
    output = open('dataset_stats.txt', 'w')
    rootdir = "datasets/RAVDESS"
    total, mean_length, min_length, max_length = 0, 0, 0, 0
    length = []
    subdirs = list(filter(lambda x: "Actor" in x, os.listdir(rootdir)))
    for i, subdir in enumerate(subdirs):
        files = list(filter(lambda x: "wav" in x, os.listdir(f"{rootdir}/{subdir}")))
        for j, file_ in enumerate(files):
            print(f"\rProcessing Actor {(i+1):02d}/{len(subdirs)}, File {(j+1):02d}/{len(files)}", end="")
            audio = librosa.load(f"{rootdir}/{subdir}/{file_}", sr=SR)[0]
            length.append(len(librosa.effects.trim(audio)[0]))
            total += 1
    mean_length = sum(length)//len(length)
    min_length = min(length)
    max_length = max(length)
    print(f"\nFinish Processing Dataset, mean_length={mean_length}, min_length={min_length}, max_length={max_length}")
    output.write(f"mean_length={mean_length}, min_length={min_length}, max_length={max_length}")
    return mean_length, min_length, max_length

def preprocess_dataset():
    rootdir = "datasets/RAVDESS"
    outputdir = "datasets/Preprocessed"
    subdirs = list(filter(lambda x: "Actor" in x, os.listdir(rootdir)))
    for i, subdir in enumerate(subdirs):
        files = list(filter(lambda x: "wav" in x, os.listdir(f"{rootdir}/{subdir}")))
        for j, file_ in enumerate(files):
            print(f"\rProcessing Actor {(i+1):02d}/{len(subdirs)}, File {(j+1):02d}/{len(files)}", end="")
            audio = librosa.load(f"{rootdir}/{subdir}/{file_}", sr=SR)[0]
            audio = librosa.effects.trim(audio)[0]
            padding = WAV_LENGTH - len(audio)
            audio = np.concatenate((np.zeros(padding//2), audio, np.zeros(padding-padding//2)))
            assert len(audio) == WAV_LENGTH
            librosa.output.write_wav(f"{outputdir}/{file_}", audio, sr=SR)
    print(f"\nFinish Processing Dataset")
    return

def get_feature_vector_and_label(file_name):
    label = int(file_name[:-4].split("-")[2])
    audio = librosa.load(f"datasets/Preprocessed/{file_name}", sr=SR)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13, hop_length=HOP_LENGTH)
    return label, mfcc

def extract_features_and_labels():
    rootdir="datasets/Preprocessed"
    files = list(filter(lambda x: "wav" in x, os.listdir("datasets/Preprocessed")))
    features = []
    labels = []
    for i, file_ in enumerate(files):
        print(f"\rProcessing file {(i+1)}/{len(files)}", end="")
        label, feature = get_feature_vector_and_label(file_)
        labels.append(label)
        features.append(feature)
    labels = np.array(labels)
    features = np.array(features)
    print(labels.shape)
    print(features.shape)
    print(f"\nFinish Extracting Features")
    return labels, features

if __name__ == '__main__':
    extract_features_and_labels()