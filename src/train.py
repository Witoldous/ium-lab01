from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Zadanie 3: wczytanie danych jako DataFrame ---
iris = load_iris(as_frame=True)
df = iris.frame                      # cechy + kolumna 'target'

print(df.head())                     # pierwsze wiersze
print("Rozmiar:", df.shape)          # (150, 5)
print(df.describe())                 # statystyki opisowe

# --- Zadanie 4: podział 80/20 i trening ---
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)          # uczymy tylko na zbiorze treningowym

# --- Zadanie 5: ocena na zbiorze testowym ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"Accuracy: {acc:.4f}")
print(report)

with open("wyniki.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n{report}")