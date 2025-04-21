from utils import load_data
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

X, y = load_data('data/', feature_type='mfcc')
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

print("SVM Classification Report:\n", classification_report(y_test, clf.predict(X_test)))

joblib.dump((clf, le), 'models/svm_model.joblib')
