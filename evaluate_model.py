import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Use absolute paths
model_path = r"D:\talk-sense\models\sentiment_model.pkl"
x_test_path = r"D:\talk-sense\data\processed\X_test.pkl"
y_test_path = r"D:\talk-sense\data\processed\y_test.csv"
report_path = r"D:\talk-sense\reports\metrics_report.md"

# Load model and test data
with open(model_path, 'rb') as f:
    clf = pickle.load(f)
with open(x_test_path, 'rb') as f:
    X_test = pickle.load(f)
y_test = pd.read_csv(y_test_path).values.ravel()

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save report
with open(report_path, 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("✅ Evaluation complete! Report saved to:", report_path)
