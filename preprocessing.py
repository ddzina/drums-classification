import json
import numpy as np
import sklearn
import librosa
import glob
import os

PATH = 'Dataset'
JSON_PATH = 'myData.json'
TARGET_SR = 16000


def detect_leading_silence(sound, silence_threshold=.001) -> int:
    """ Normalization and trimming the silence

    :return: start ms of the sound
    """

    trim_ms = 0
    max_num = max(sound)
    sound = sound/max_num
    sound = np.array(sound)
    for i in range(len(sound)):
        if sound[trim_ms] < silence_threshold:
            trim_ms += 1
    return trim_ms


def feature_extract() -> dict[str, list]:
    """
    Extracts mfcc vectors from audio data and labels the data

    :return: dict with mapping names, mfcc vectors and labels
    """
    sr = 44100
    window_size = 512
    hop_size = 256
    data = {
        'mapping': [],
        'mfcc': [],
        'labels': []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(PATH)):

        if dirpath is not PATH:
            dirpath_components = dirpath.split('\\')
            semantic_label = dirpath_components[-1]
            data['mapping'].append(semantic_label)
            print('\nProcessing {}'.format(semantic_label))

            for filename in filenames:
                # resampling and normalizing audio

                music, sr = librosa.load(os.path.join(dirpath, filename), sr=sr)
                music = librosa.resample(music, orig_sr=sr, target_sr=TARGET_SR)
                start_trim = detect_leading_silence(music)
                end_trim = detect_leading_silence(np.flipud(music))

                duration = len(music)
                trimmed_sound = music[start_trim:duration-end_trim]

                # use mfcc to calculate the audio features

                mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=TARGET_SR, n_mfcc=20,
                                             hop_length=hop_size, n_fft=window_size)
                aver = np.mean(mfccs, axis=1)
                feature = aver.reshape(20)

                # store label and feature

                data['mfcc'].append(feature.tolist())
                data['labels'].append(i-1)
                print(filename, i-1)
                
    return data


def main():
    """ Extract features and save them to json file """
    data = feature_extract()
    with open(JSON_PATH, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    main()
