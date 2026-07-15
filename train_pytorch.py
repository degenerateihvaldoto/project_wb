import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# НАСТРОЙКИ
MODEL_NAME = "cointegrated/rubert-tiny2"
DATA_PATH = "data/processed/reviews_balanced.csv"
SAVE_MODEL_DIR = "models/pytorch_bert" # Сюда сохраним обученную модель
BATCH_SIZE = 16
EPOCHS = 2  # Для начала 2 эпох вполне хватит, чтобы не ждать долго
LEARNING_RATE = 2e-5

# Выбираем устройство (GPU если есть, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# ПОДГОТОВКА ДАННЫХ И PYTORCH DATASET
class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.values
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Токенизация текста под формат BERT
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Загружаем данные и делим так же, как в бейзлайне
df = pd.read_csv(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Загружаем токенайзер
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Создаем PyTorch датасеты
train_dataset = ReviewsDataset(X_train, y_train, tokenizer)
test_dataset = ReviewsDataset(X_test, y_test, tokenizer)

# Создаем DataLoader'ы для батчевания
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ, ЛОССА И ОПТИМИЗАТОРА
# Загружаем предобученную модель для классификации на 2 класса (0 и 1)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# ЦИКЛ ОБУЧЕНИЯ
print("\nНачинаем обучение нейросети на PyTorch...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    # Визуализируем прогресс с помощью tqdm
    loop = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS}")
    for batch in loop:
        optimizer.zero_grad()
        
        # Переносим батч на GPU/CPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Прямой проход
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Обратный проход и шаг оптимизатора
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"Средний Loss на обучении: {avg_train_loss:.4f}")

# ОЦЕНКА МОДЕЛИ (EVALUATION)
print("\nОцениваем модель на тестовой выборке...")
model.eval()

all_preds = []
all_labels = []

with torch.no_grad(): # Отключаем расчет градиентов для ускорения
    for batch in tqdm(test_loader, desc="Валидация"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Получаем предсказанный класс (0 или 1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Считаем итоговые метрики
final_acc = accuracy_score(all_labels, all_preds)
final_f1 = f1_score(all_labels, all_preds)

print("\n=== Результаты PyTorch модели ===")
print(f"Accuracy: {final_acc:.4f} (Было на Baseline: 0.9022)")
print(f"F1-Score: {final_f1:.4f} (Было на Baseline: 0.8988)")

# Сохраняем модель
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
model.save_pretrained(SAVE_MODEL_DIR)
tokenizer.save_pretrained(SAVE_MODEL_DIR)
print(f"\nМодель успешно сохранена в папку {SAVE_MODEL_DIR}!")