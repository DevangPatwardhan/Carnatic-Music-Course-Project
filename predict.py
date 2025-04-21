import joblib
import librosa
from utils import extract_mfcc
import numpy as np
from tensorflow.keras.models import load_model

def predict_with_svm(file_path):
    clf, le = joblib.load('models/svm_model.joblib')
    feat = extract_mfcc(file_path)
    pred = clf.predict([feat])
    print("SVM Prediction:", le.inverse_transform(pred)[0])

def predict_with_cnn(file_path):
    model = load_model('models/cnn_spectrogram.h5')
    img = extract_spectrogram(file_path)
    img = img.reshape(1, 180, 180, 1)
    pred = np.argmax(model.predict(img), axis=1)
    labels = ['Shankarabharanam', 'Kalyani', 'Kharaharapriya', 'Todi', 'Bhairavi']
    print("CNN Prediction:", labels[pred[0]])

# Example Usage
predict_with_svm("data/Bhairavi/sample1.wav")
predict_with_cnn("data/Bhairavi/sample1.wav")

