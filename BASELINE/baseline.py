import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Or any other classifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X, y = data.data, data.target

df = pd.DataFrame(X, columns=data.feature_names)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


df['target'] = y
print("--- Basic Statistics ---")
print(df.describe())
print("\n--- Class Distribution ---")
print(df['target'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the features
scaler = StandardScaler()
# We "fit" the scaler ONLY on the training data to learn the mean and variance,
# and then we transform the training data.
X_train_scaled = scaler.fit_transform(X_train)
# We transform the test data using the exact same scale learned from the training data.
X_test_scaled = scaler.transform(X_test)

# 4. Train the model (Using Logistic Regression to see the impact of scaling)
# max_iter is increased slightly because scaled data can sometimes take a bit longer to converge
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))