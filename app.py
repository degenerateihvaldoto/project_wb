import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Настраиваем страницу
st.set_page_config(page_title="WB Sentiment Analyzer", page_icon="🛍️")

st.title("🛍️ Анализатор отзывов Wildberries")
st.write("Вставьте текст отзыва ниже, и нейросеть на PyTorch определит, качественный ли это товар.")

# Кэшируем загрузку модели, чтобы страница не зависала при каждом клике
@st.cache_resource
def load_model():
    model_path = "models/pytorch_bert"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_model()
    model.eval()
    st.success("Модель PyTorch успешно загружена!")
except Exception as e:
    st.error(f"Не удалось загрузить модель. Убедитесь, что она сохранена в models/pytorch_bert. Ошибка: {e}")

# Поле для ввода текста
user_input = st.text_area("Текст отзыва:", placeholder="Например: Товар пришел быстро, качество супер, нитки не торчат!")

if st.button("Проанализировать качество"):
    if user_input.strip() == "":
        st.warning("Пожалуйста, введите текст отзыва.")
    else:
        # Токенизация и предсказание
        inputs = tokenizer(user_input, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).flatten()
            pred_class = torch.argmax(logits, dim=1).item()
        
        # Вывод результатов
        prob_neg = probs[0].item() * 100
        prob_pos = probs[1].item() * 100
        
        st.write("---")
        if pred_class == 1:
            st.markdown(f"### 🎉 Вердикт: **Товар хороший (Норм)**")
            st.write(f"Уверенность модели: {prob_pos:.2f}%")
        else:
            st.markdown(f"### ❌ Вердикт: **Товар плохой (Брак / Негатив)**")
            st.write(f"Уверенность модели: {prob_neg:.2f}%")
            
        # Рисуем прогресс-бары для наглядности
        st.write("Вероятности классов:")
        st.progress(probs[1].item(), text=f"Позитив: {prob_pos:.1f}%")
        st.progress(probs[0].item(), text=f"Негатив: {prob_neg:.1f}%")