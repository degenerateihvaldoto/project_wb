import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_PATH = "models/pytorch_bert"
DATA_PATH = "data/processed/reviews_balanced.csv" #путь сохранения нашего спаршшеного цсв файла

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель и токенайзер
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Загружаем тестовые данные (точно такие же, как при обучении)
df = pd.read_csv(DATA_PATH)
_, X_test, _, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

test_df = pd.DataFrame({'text': X_test, 'label': y_test}).reset_index(drop=True)

print("Ищем ошибки модели...")
errors = []

with torch.no_grad():
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        text = str(row['text'])
        true_label = row['label']
        
        # Токенизация
        inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=1).item()
        
        # Если модель ошиблась, сохраняем пример
        if pred_label != true_label:
            errors.append({
                'text': text,
                'true': true_label,
                'pred': pred_label
            })
            if len(errors) >= 10: # Хватит 10 примеров для анализа
                break

print("\n=== ПРИМЕРЫ ОШИБОК МОДЕЛИ ===")
for i, err in enumerate(errors):
    print(f"\nПример {i+1}:")
    print(f"Текст отзыва: {err['text']}")
    print(f"Реальный класс (из звезд): {err['true']} | Предсказание модели: {err['pred']}")