# Drum classification using machine learning model

## Description:

This project is my research work at the university.
It consists of a few files:
* **rename_script.py** includes script that renames the audio files in a dataset folder, files were collected form different
  sources, so I had to rename them to preprocess.
* **preprocessing.py** includes the preprocessing algorithm that extracts mfcc vectors from audio data that will be an
  input for the model.
* **SVM_Classifier.ipynb** includes the functions for training and evaluating the model of SVM.

Project takes the next steps:
1. Collecting data and forming the dataset.
2. Feature extraction.
3. SVM model training.
4. Model evaluation using K-Fold.

## Dataset:

Dataset contains 3000 .wav files - 500 samples for each instrument:
* Kick
* Bass (basically not a drum, but I included it, since it has close frequency range to kick, in order to
test abilities of classifier)
* Snare
* Crash
* HiHat
* Clap

I had an experience of working as a sound engineer, so I have a lot of sound libraries collected, which include
various drum samples. So I had no problem finding sources and collecting data.


## Development tools:
* Python 3.10
* Scikit-learn
* Librosa
* Matplotlib
* Numpy

## Results:
Evaluation results stored in the log.txt file:

                precision   recall   f1-score   support

        Bass       0.89      0.83      0.86        59
        Clap       0.84      0.98      0.91        49
       Crash       0.76      0.74      0.75        47
       HiHat       0.83      0.76      0.80        46
        Kick       0.79      0.82      0.81        51
       Snare       0.87      0.85      0.86        48

    accuracy                           0.83       300
    macro avg      0.83      0.83      0.83       300
    weighted avg   0.83      0.83      0.83       300

Overall, I've got satisfactory scores for my first project. There are few classes that were predicted worse than others,
so I have a field for future experiments enhancing my classifier.

## Installation:
### 1. Clone the repo
    git clone https://github.com/ddzina/drums-classification.git

### 2. Create Anaconda virtual environment:

    conda create --name env --file requirements.txt

Put your name of environment after flag "--name". Created environment will include all the requirement dependencies.




## Roadmap:
1. Increase the amount of collected data to get better scores.
2. Try RandomForest model and compare to SVM.
3. Add more instruments including piano, guitar and flute.

