import librosa
import numpy as np
import cv2
import os

def extract_mfcc(path, n_mfcc=20):
    y, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def extract_spectrogram(path, size=(180, 180)):
    y, sr = librosa.load(path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return cv2.resize(mel_db, size)

def load_data(data_path, feature_type='mfcc', flatten=True):
    X, y = [], []
    for label in os.listdir(data_path):
        folder = os.path.join(data_path, label)
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            if feature_type == 'mfcc':
                features = extract_mfcc(filepath)
            else:
                features = extract_spectrogram(filepath)
                if flatten:
                    features = features.flatten()
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

