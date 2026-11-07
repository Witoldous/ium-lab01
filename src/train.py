import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie konfiguracji z pliku .env (wartości domyślne na wypadek braku klucza)
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/iris.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Jednorazowo zapisujemy zbiór iris do CSV pod ścieżką z .env
if not os.path.exists(DATA_PATH):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    load_iris(as_frame=True).frame.to_csv(DATA_PATH, index=False)

df = pd.read_csv(DATA_PATH)
print(df.head())
print("Rozmiar:", df.shape)
print(df.describe())

X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=load_iris().target_names)

print(f"Accuracy: {acc:.4f}")
print(report)

with open("wyniki.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n{report}")