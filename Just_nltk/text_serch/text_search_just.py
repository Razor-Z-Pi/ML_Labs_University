import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import re
import random

# Словарь
arr = [
    "хорqwdоший", "дело", "бегать", "прыгать",
    "красивый", "добрый", "классный", "могучий",
    "умный", "еда", "крутой", "игра",
    "программист", "разработчик", "жизнь", "хлеб",
    "гора", "красный", "синий", "белый",
    "лес", "поля", "степь", "хакер",
    "мужчина", "женщина", "студент", "ученик",
    "учитель", "машина",
]

# Создаем словарь для кодирования слов
word_to_idx = {word: idx for idx, word in enumerate(arr)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(arr)

# Параметры модели
embedding_dim = 50
hidden_dim = 128
output_dim = vocab_size

class WordClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(WordClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        # Усредняем embedding'и для всего предложения
        pooled = torch.mean(embedded, dim = 1)
        hidden = self.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        return output

# Создаем модель
model = WordClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# Функция для предобработки текста
def preprocess_text(text):
    # Приводим к нижнему регистру и разбиваем на слова
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# Функция для кодирования предложения
def encode_sentence(sentence, word_to_idx):
    words = preprocess_text(sentence)
    indices = []
    for word in words:
        if word in word_to_idx:
            indices.append(word_to_idx[word])
        else:
            # Для неизвестных слов используем случайный индекс
            indices.append(random.randint(0, vocab_size - 1))
    return torch.tensor(indices, dtype=torch.long)

# Создаем обучающий датасет
def create_training_dataset():
    dataset = []
    
    # Примеры для каждого слова из словаря
    examples_per_word = 20
    
    for word in arr:
        # Создаем различные контексты для каждого слова
        contexts = [
            f"Я очень {word} сегодня",
            f"Это {word} и интересно",
            f"Мне нравится {word}",
            f"{word.capitalize()} - это здорово",
            f"Мы будем {word} вместе",
            f"Ты очень {word}",
            f"Он любит {word}",
            f"Она занимается {word}",
            f"Этот {word} прекрасен",
            f"Мой друг - {word}",
            f"Я хочу стать {word}",
            f"Мы изучаем {word}",
            f"Это похоже на {word}",
            f"Я мечтаю о {word}",
            f"Они практикуют {word}",
            f"Этот человек - {word}",
            f"Я наслаждаюсь {word}",
            f"Мы создаем {word}",
            f"Он объясняет {word}",
            f"Она демонстрирует {word}"
        ]
        
        for context in contexts[:examples_per_word]:
            dataset.append((context, word))
    
    # Добавляем немного негативных примеров
    negative_examples = [
        ("Сегодня хорошая погода", "погода"),
        ("Я иду в магазин", "магазин"),
        ("Он читает книгу", "книга"),
        ("Мы смотрим фильм", "фильм"),
        ("Она пишет письмо", "письмо")
    ]
    
    for context, word in negative_examples:
        if word not in word_to_idx:
            dataset.append((context, random.choice(arr)))
    
    return dataset

def train_model(model, dataset, word_to_idx, epochs = 100, batch_size = 32, learning_rate = 0.003):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)
    
    # Перемешиваем данные
    random.shuffle(dataset)
    
    # Разделяем на батчи
    batches = []
    for i in range(0, len(dataset), batch_size):
        batches.append(dataset[i:i + batch_size])
    
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in batches:
            batch_loss = 0
            optimizer.zero_grad()
            
            for sentence, target_word in batch:
                # Пропускаем если target_word нет в словаре
                if target_word not in word_to_idx:
                    continue
                
                # Подготовка данных
                input_indices = encode_sentence(sentence, word_to_idx)
                target_idx = torch.tensor([word_to_idx[target_word]], dtype=torch.long)
                
                if len(input_indices) == 0:
                    continue
                
                # Forward pass
                outputs = model(input_indices.unsqueeze(0))
                loss = criterion(outputs, target_idx)
                batch_loss += loss
              
                _, predicted = torch.max(outputs, 1)
                if predicted.item() == target_idx.item():
                    correct += 1
                total += 1
            
            # Backward pass для батча
            if len(batch) > 0:
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
        
        scheduler.step()
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(batches) if batches else 0
        
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Эпох [{epoch + 1} / {epochs}], Потери: {avg_loss:.4f}, Точность: {accuracy:.4f}')
    
    return losses

# Функция для предсказания
def predict_words(sentence, model, word_to_idx, arr):
    model.eval()
    with torch.no_grad():
        # Кодируем предложение
        input_indices = encode_sentence(sentence, word_to_idx)
        if len(input_indices) == 0:
            return "Нет слов для классификации!!!"
        
        input_tensor = input_indices.unsqueeze(0)
        
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim = 1)
        max_prob, predicted = torch.max(probabilities, 1)
        
        # Получаем наиболее вероятное слово из словаря
        predicted_word = arr[predicted.item()]
        confidence = max_prob.item()
        
        return predicted_word, confidence

# Функция для получения топ-N предсказаний
def get_top_predictions(sentence, model, word_to_idx, arr, top_n=5):
    model.eval()
    with torch.no_grad():
        input_indices = encode_sentence(sentence, word_to_idx)
        if len(input_indices) == 0:
            return []
        
        input_tensor = input_indices.unsqueeze(0)
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim = 1)
        
        # Получаем топ-N предсказаний
        probs, indices = torch.topk(probabilities, top_n, dim = 1)
        
        results = []
        for i in range(top_n):
            word = arr[indices[0][i].item()]
            prob = probs[0][i].item()
            results.append((word, prob))
        
        return results

def main():
    print("Создание датасета...")
    dataset = create_training_dataset()
    print(f"Размер датасета: {len(dataset)} примеров")
    
    print("\nНачало обучения...")
    losses = train_model(model, dataset, word_to_idx, epochs = 300, batch_size = 32)
    
    print("\nОбучение завершено!!!")
    
    # Тестируем модель
    test_sentences = [
        "Я люблю программировать и учиться",
        "Он очень умный студент",
        "Мы будем бегать утром",
        "Эта машина очень быстрая",
        "Красивый лес осенью",
        "Она добрая женщина",
        "Хлеб свежий и вкусный",
        "Синий цвет неба"
    ]
    
    print("\nТестирование модели:")
    print("-" * 50)
    
    for sentence in test_sentences:
        predicted_word, confidence = predict_words(sentence, model, word_to_idx, arr)
        print(f"Вход: '{sentence}'")
        print(f"Предсказание: '{predicted_word}' (уверенность: {confidence:.4f})")
        
        # Покажем топ-3 предсказания
        top_predictions = get_top_predictions(sentence, model, word_to_idx, arr, 3)
        print("Топ-3 варианта:")
        for i, (word, prob) in enumerate(top_predictions, 1):
            print(f"  {i} . {word}: {prob:.4f}")
        print("-" * 30)

main()
     
print("\nМодель готова к использованию!!!")