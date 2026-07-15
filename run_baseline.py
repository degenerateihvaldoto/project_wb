import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Загружаем наши подготовленные сбалансированные данные
DATA_PATH = "data/processed/reviews_balanced.csv"
df = pd.read_csv(DATA_PATH)

print(f"Загружено строк: {len(df)}")

# Делим данные на обучающую (train) и тестовую (test) выборки (80% на 20%)
# random_state нужен для воспроизводимости результатов (чтобы при каждом запуске деление было одинаковым)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label'] # сохраняем баланс классов в train и test
)

print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# Векторизуем текст с помощью TF-IDF
# Ограничимся 10 000 самых частых слов/сочетаний слов
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучаем простую модель классического ML - Логистическую регрессию
print("\nОбучаем Baseline (Logistic Regression)...")
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_vec, y_train)

# Делаем предсказания на тестовых данных
y_pred = baseline_model.predict(X_test_vec)

# Считаем метрики (Пункт "Метрики" из твоего чеклиста!)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Результаты Baseline ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nДетальный отчет по классам:")
print(classification_report(y_test, y_pred, target_names=['Негатив (0)', 'Позитив (1)']))