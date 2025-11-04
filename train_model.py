import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

with open(r"D:\talk-sense\data\processed\X_train.pkl", 'rb') as f:
    X_train = pickle.load(f)

y_train = pd.read_csv(r"D:\talk-sense\data\processed\y_train.csv").values.ravel()

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

with open(r"D:\talk-sense\models\sentiment_model.pkl", 'wb') as f:
    pickle.dump(clf, f)

print("✅ Model trained and saved successfully!")
