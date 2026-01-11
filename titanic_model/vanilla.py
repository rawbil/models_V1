import joblib
import numpy as np

model = joblib.load("titanic_model.pkl")
X_new = np.array([[1, 0, 22.0, 7.2500, 1, 0, 0, 1, 0, 0, 1]])
X_new = np.array([[1, 1, 0, 38.0, 71.2833, 0, 1, 0, 0, 1, 0]])
X_new = np.array([[0, 0, 26.0, 7.9250, 0, 0, 0, 1, 0, 0, 1]])
X_new = np.array([[0, 0, 35.0, 8.0500, 1, 0, 0, 1, 0, 0, 1]])
# X_new = np.reshape(X_new, (-1, 1))

predict = model.predict(X_new)
probability = model.predict_proba(X_new)
print(predict[0])
print(probability[0][1])
