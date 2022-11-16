import json

import numpy as np
import sklearn
import librosa
import glob
import os

PATH = 'Dataset'
JSON_PATH = 'myData.json'

def detect_leading_silence(sound, silence_threshold=.001, chunk_size=10):
    # this function first normalizes audio data
    #calculates the amplitude of each frame
    #silence_threshold is used to flip the silence part
    #the number of silence frame is returned.
    #trim_ms is the counter
    trim_ms = 0
    max_num = max(sound)
    sound = sound/max_num
    sound = np.array(sound)
    for i in range(len(sound)):
        if sound[trim_ms] < silence_threshold:
            trim_ms += 1
    return trim_ms


def feature_extract():

    sr = 44100
    window_size = 4096
    hop_size = window_size/2
    data = {
        'mapping': [],
        'mfcc': [],
        'labels': []
    }
    #read file
    #files = glob.glob('Dataset/*/*.wav')
    #np.random.shuffle(files)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(PATH)):
        #np.random.shuffle(filenames)

        if dirpath is not PATH:
            dirpath_components = dirpath.split('\\')
            semantic_label = dirpath_components[-1]
            data['mapping'].append(semantic_label)
            print('\nProcessing {}'.format(semantic_label))
            for filename in filenames:

                music, sr = librosa.load(os.path.join(dirpath, filename), sr=sr)

                start_trim = detect_leading_silence(music)
                end_trim = detect_leading_silence(np.flipud(music))

                duration = len(music)
                trimmed_sound = music[start_trim:duration-end_trim]
                # the sound without silence

                #use mfcc to calculate the audio features
                mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr, n_fft=window_size)
                aver = np.mean(mfccs, axis = 1)
                feature = aver.reshape(20)

                #store label and feature
                #the output should be a list
                #label and feature, corresponds one by one
                #feature.append(aver)

                data['mfcc'].append(feature.tolist())
                data['labels'].append(i-1)
                print(filename, i-1)
                #data = np.vstack((data, data2))
                # print data
    return data


def main():
    data = feature_extract()
    with open(JSON_PATH, 'w') as fp:
        json.dump(data, fp, indent=4)
    print(data)
    print(len(data))


if __name__ == '__main__':
    main()
