import zstandard as zstd
import os
import json
import io
import pandas as pd

# Настройки
INPUT_FILE = "feedbacks-02.json.zst"
OUTPUT_FILE = "data/processed/reviews_balanced.csv" # Сохраним сюда
TARGET_SIZE_PER_CLASS = 10000  # Соберем по 10к каждого класса

positives = []
negatives = []

print("Начинаем обработку архива...")

with open(INPUT_FILE, "rb") as f:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        
        for i, line in enumerate(text_stream):
            try:
                review = json.loads(line)
                text = review.get('text', '').strip()
                rating = review.get('productValuation')
                
                # Пропускаем пустые отзывы
                if not text or rating is None:
                    continue
                
                # Классифицируем по звездам
                if rating in [1, 2, 3]:
                    # Негатив (Класс 0)
                    if len(negatives) < TARGET_SIZE_PER_CLASS:
                        negatives.append({'text': text, 'label': 0, 'stars': rating})
                elif rating in [4, 5]:
                    # Позитив (Класс 1)
                    if len(positives) < TARGET_SIZE_PER_CLASS:
                        positives.append({'text': text, 'label': 1, 'stars': rating})
                
                # Если набрали нужное количество — останавливаемся
                if len(negatives) >= TARGET_SIZE_PER_CLASS and len(positives) >= TARGET_SIZE_PER_CLASS:
                    print(f"Успешно собрали по {TARGET_SIZE_PER_CLASS} примеров каждого класса!")
                    break
                    
                # Просто лог каждые 100 000 строк, чтобы видеть прогресс
                if i % 100000 == 0 and i > 0:
                    print(f"Обработано {i} строк. Собрано негатива: {len(negatives)}, позитива: {len(positives)}")
                    
            except Exception as e:
                # если будут битыестроки в json
                continue

# Объединяем в один датасет
all_reviews = negatives + positives

# Переводим в Pandas DataFrame и перемешиваем, чтобы классы не шли подряд
df = pd.DataFrame(all_reviews)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Проверим, что получилось
print("\nРаспределение классов в итоговом датасете:")
print(df['label'].value_counts())

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"\nСбалансированный датасет сохранен в: {OUTPUT_FILE}")