import os, librosa

def open_file_with_label():
    audio = []
    label = []
    rootdir = "datasets/RAVDESS"
    actors = list(filter(lambda x: "Actor" in x, os.listdir(rootdir)))
    print("Start Loading Dataset")
    for i, actor in enumerate(actors):
        wav_files = list(filter(lambda x: "wav" in x, os.listdir(f"{rootdir}/{actor}")))
        for j, wav_file in enumerate(wav_files):
            print(f"\r  Actor {(i+1):02d}/{len(actors)}, File {(j+1):02d}/{len(wav_files)}", end="")
            audio.append(librosa.load(f"{rootdir}/{actor}/{wav_file}")[0])
            label.append([int(i) for i in wav_file[:-4].split("-")])
    print("\nFinish Loading Dataset")
    return audio, label

open_file_with_label()