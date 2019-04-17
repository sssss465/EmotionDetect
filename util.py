import os, librosa
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

def get_dataset_stats():
    rootdir = "datasets/RAVDESS"
    sr, total, mean_length, min_length, max_length = 0, 0, 0, 0, 0
    length = []
    subdirs = list(filter(lambda x: "Actor" in x, os.listdir(rootdir)))
    for i, subdir in enumerate(subdirs):
        files = list(filter(lambda x: "wav" in x, os.listdir(f"{rootdir}/{subdir}")))
        for j, file_ in enumerate(files):
            print(f"\rProcessing Actor {(i+1):02d}/{len(subdirs)}, File {(j+1):02d}/{len(files)}", end="")
            audio = librosa.load(f"{rootdir}/{subdir}/{file_}")
            if sr == 0:
                sr = audio[1]
            length.append(len(librosa.effects.trim(audio[0])[0]))
            total += 1
    mean_length = sum(length)/len(length)
    min_length = min(length)
    max_length = max(length)
    print(f"\nFinish Processing Dataset, sr={sr} , mean_length={mean_length}, min_length={min_length}, max_length={max_length}")
    return sr, mean_length, min_length, max_length

def get_mfcc(audio, hop=20, window=12):
    return

def get_pitch(audio, windows=12):
    return

    

if __name__ == '__main__':
    print(get_dataset_stats())