import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
import random
import matplotlib.pyplot as plt
from collections import Counter
import re
import math
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import time

# Проект опираеться на доки pytorch

class HighPrecisionTextGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_corpus_data(self):
        """Расширенный и разнообразный корпус текста"""
        print("Загрузка расширенного корпуса текстов...")
        
        extended_text = """
        заяц бежал по лесной тропинке и внезапно увидел рыжую лису которая сидела под большим дубом 
        испуганный заяц быстро развернулся и побежал в другую сторону потому что боялся хитрой лисы 
        в густом лесу жили разные животные белки птицы и другие зайцы которые прятались от опасности 
        заяц бежал очень быстро чтобы спастись и вскоре нашел свою глубокую нору где можно было укрыться 
        
        пушистый кот сидел на подоконнике и смотрел на улицу где светило яркое солнце и пели птицы 
        большое дерево стояло посреди зеленого поля и его ветви качались на свежем ветру 
        мальчик читал интересную книгу в саду и не замечал как проходит время 
        девочка рисовала красивую картину красками и у нее получалось очень хорошо 
        
        собака бежала за мячом по траве и легко его ловила потому что была быстрой 
        учитель объяснял урок ученикам которые внимательно слушали и задавали вопросы 
        
        река текла через долину и вода в ней была чистой и прозрачной 
        старый дом стоял на холме и в нем жила дружная семья 
        утреннее солнце освещало поляну и роса блестела на зеленой траве 
        
        летом в лесу растут грибы и ягоды которые собирают люди 
        зимой выпадает снег и дети катаются на санках с горки 
        весной природа просыпается и появляются первые цветы 
        осенью листья желтеют и опадают с деревьев 
        
        тучи закрывали небо и начинался дождь поэтому люди прятались под навесами 
        вечером на небе появлялись звезды и луна а город зажигал огни 
        """
        
        full_text = extended_text
        full_text = self.clean_text(full_text)

        word_count = len(full_text.split())
        print(f"Загружено: {len(full_text)} символов, {word_count} слов")
        print(f"Уникальных слов: {len(set(full_text.split()))}")
        
        return full_text.lower()

    def clean_text(self, text):
        """Тщательная очистка текста, в основном от точек"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\?\!]', '', text)
        # Убираем лишние точки и знаки препинания
        text = re.sub(r'\.{2,}', '.', text)
        return text

    def preprocess_text(self, text, seq_length = 10, min_freq = 1):
        """Улучшенная предобработка с анализом данных"""
        print("Токенизация текста...(✿◡‿◡)")

        tokens = word_tokenize(text)
        print(f"Всего токенов: {len(tokens)}")

        word_freq = Counter(tokens)
        print(f"Самые частые слова: {word_freq.most_common(10)}")
        
        meaningful_tokens = [word for word in tokens if word_freq[word] >= min_freq and word not in ['.', ',', '!', '?']]
        filtered_tokens = meaningful_tokens
        
        print(f"Токенов после фильтрации: {len(filtered_tokens)}")

        self.vocab = ['<UNK>', '<START>', '<END>', '<PAD>'] + sorted(set(filtered_tokens))
        self.vocab_size = len(self.vocab)

        print(f"Размер словаря: {self.vocab_size}")

        # Создаем mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

        print("Создание последовательностей...")
        encoded_text = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                       for word in tokens if word in self.word_to_idx]

        sequences = []
        targets = []
        
        step_size = 2  # Увеличиваем шаг для разнообразия данных

        for i in range(0, len(encoded_text) - seq_length, step_size):
            seq = encoded_text[i:i + seq_length]
            target = encoded_text[i + seq_length]
            sequences.append(seq)
            targets.append(target)

        print(f"Создано последовательностей: {len(sequences)}")
        
        # Разделяем на train/validation
        X_train, X_val, y_train, y_val = train_test_split(sequences, targets, test_size = 0.15, random_state = 42)
        
        print(f"Тренировочные данные: {len(X_train)}")
        print(f"Валидационные данные: {len(X_val)}")
        
        return (np.array(X_train), np.array(X_val), 
                np.array(y_train), np.array(y_val))

class ImprovedLSTMModel(nn.Module): # Моделька
    def __init__(self, vocab_size, embedding_dim = 128, hidden_dim = 256, n_layers = 2, dropout = 0.3):
        super(ImprovedLSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 3)
        
        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout, batch_first = True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Инициализация forget gate
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Embedding
        x = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Берем только последний выход
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Классификация
        output = self.classifier(lstm_out)
        
        return output, hidden

class ImprovedTrainer:
    def __init__(self, model, train_loader, val_loader, device, word_to_idx, idx_to_word):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

        # Оптимизатор
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr = 0.002,
            weight_decay = 1e-5
        )
        
        # Планировщик
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma = 0.8)
        
        # Критерий
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.best_val_loss = float('inf')
        self.early_stop_patience = 10
        self.early_stop_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Прямой проход
            output, _ = self.model(data)
            loss = self.criterion(output, targets)

            # Обратный проход
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 0.5)
            
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            
            # Вычисление точности
            _, predicted = torch.max(output, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                accuracy = correct / total if total > 0 else 0
                print(f'Эпоха: {epoch:2d}, Батч: {batch_idx:3d}, Потери: {loss.item():.4f}, '
                      f'Точность: {accuracy:.4f}, LR: {current_lr:.6f}')

        avg_loss = total_loss / batch_count
        accuracy = correct / total if total > 0 else 0
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        epoch_time = time.time() - start_time
        print(f'Эпоха {epoch} завершена за {epoch_time:.1f}с. Потери: {avg_loss:.4f}, Точность: {accuracy:.4f}')
        
        return avg_loss, accuracy

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                output, _ = self.model(data)
                loss = self.criterion(output, targets)
                
                total_loss += loss.item()
                batch_count += 1
                
                _, predicted = torch.max(output, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / batch_count
        accuracy = correct / total if total > 0 else 0
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        print(f'Валидация эпохи {epoch}: Потери: {avg_loss:.4f}, Точность: {accuracy:.4f}')

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            print(f"Ранняя остановка, счетчик: {self.early_stop_counter} / {self.early_stop_patience}")
        
        return avg_loss

    def train(self, epochs=50):
        print("Начало обучения...O(*￣▽￣*)o")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch + 1)
            val_loss = self.validate(epoch + 1)
            
            # Обновление скорость обучения
            self.scheduler.step()
            
            # Генерация примера текста
            if (epoch + 1) % 2 == 0:
                self.generate_example(epoch + 1)

            if self.early_stop_counter >= self.early_stop_patience:
                print(f"Остановка на эпохе {epoch + 1}")
                break

    def generate_example(self, epoch):
        start_texts = ["заяц бежал", "кот сидел", "солнце светило", "мальчик читал"]
        
        for start_text in start_texts:
            generated = self.generate_text(start_text, length = 12, temperature = 0.8)
            print(f"Эпоха {epoch}, начало '{start_text}': {generated}")

    def generate_text(self, start_text, length = 15, temperature = 0.8):
        self.model.eval()
        
        words = word_tokenize(start_text.lower())
        encoded = []
        
        for word in words:
            if word in self.word_to_idx:
                encoded.append(self.word_to_idx[word])
            else:
                encoded.append(self.word_to_idx['<UNK>'])
        
        if not encoded:
            encoded = [random.randint(4, len(self.word_to_idx) - 1)]

        current_sequence = encoded.copy()
        generated_indices = encoded.copy()

        with torch.no_grad():
            hidden = None

            for i in range(length):
                if len(current_sequence) < 10:
                    input_seq = current_sequence + [3] * (10 - len(current_sequence))
                else:
                    input_seq = current_sequence[-10:]

                input_tensor = torch.LongTensor([input_seq]).to(self.device)

                output, hidden = self.model(input_tensor, hidden)

                if temperature > 0:
                    output = output / temperature
                    probabilities = F.softmax(output, dim=-1)
                    
                    # Фильтруем маловероятные варианты
                    probs = probabilities.cpu().numpy()[0]
                    # Убираем специальные токены из рассмотрения
                    probs[:4] = 0
                    # Нормализуем
                    probs = probs / probs.sum()
                    
                    predicted_idx = np.random.choice(len(probs), p=probs)
                else:
                    predicted_idx = torch.argmax(output, dim = -1).item()

                # Пропускаем специальные токены
                if predicted_idx < 4:
                    # Выбираем случайное слово из словаря
                    predicted_idx = random.randint(4, len(self.idx_to_word) - 1)

                generated_indices.append(predicted_idx)
                current_sequence.append(predicted_idx)
                
                # Ограничиваем длину контекста
                if len(current_sequence) > 15:
                    current_sequence = current_sequence[-10:]

        # Конвертируем обратно в текст
        generated_words = []
        for idx in generated_indices:
            if idx in self.idx_to_word and idx >= 4:
                word = self.idx_to_word[idx]
                generated_words.append(word)

        generated_text = ' '.join(generated_words)
        
        # Добавляем точку в конце
        if generated_text and not generated_text.endswith(('.', '!', '?')):
            generated_text += '.'
            
        generated_text = generated_text.capitalize()
        
        return generated_text

    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
        
        # График потерь
        ax1.plot(self.train_losses, label = 'Потери', linewidth = 2)
        ax1.plot(self.val_losses, label = 'Валидация потерь', linewidth = 2)
        ax1.set_xlabel('Эпохи')
        ax1.set_ylabel('Потери')
        ax1.set_title('Динамика потерь при обучении')
        ax1.legend()
        ax1.grid(True, alpha = 0.3)
        
        # График точности
        ax2.plot(self.train_accuracies, label = 'Точность', linewidth = 2)
        ax2.plot(self.val_accuracies, label = 'Валидация точности', linewidth = 2)
        ax2.set_xlabel('Эпохи')
        ax2.set_ylabel('Точность')
        ax2.set_title('Динамика точности при обучении')
        ax2.legend()
        ax2.grid(True, alpha = 0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    # Инициализация
    text_gen = HighPrecisionTextGenerator()

    # Загрузка и предобработка данных
    text = text_gen.load_corpus_data()
    X_train, X_val, y_train, y_val = text_gen.preprocess_text(text, seq_length = 10, min_freq = 1)

    print(f"Тренировочные данные: {len(X_train)} последовательностей")
    print(f"Валидационные данные: {len(X_val)} последовательностей")

    if len(X_train) == 0:
        print("Ошибка: не создано обучающих последовательностей!!!")
        return

    # Создание DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(X_val), torch.LongTensor(y_val)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Создание модели
    vocab_size = text_gen.vocab_size
    model = ImprovedLSTMModel( vocab_size=vocab_size, embedding_dim = 128, hidden_dim = 256, n_layers = 2, dropout = 0.3)

    print(f"Модель создана!!!")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    trainer = ImprovedTrainer(
        model, train_loader, val_loader, text_gen.device,
        text_gen.word_to_idx, text_gen.idx_to_word
    )
    
    trainer.train(epochs = 30)

    trainer.plot_training()

    test_phrases = [
        "заяц бежал",
        "кот сидел", 
        "солнце светило",
        "мальчик читал",
        "дерево стояло"
    ]

    for phrase in test_phrases:
        print(f"\nНачальная фраза: '{phrase}'")
        print("-" * 50)
        
        for temp in [0.7, 0.8, 0.9]:
            generated = trainer.generate_text(
                phrase, length=12, temperature=temp
            )
            print(f"temp={temp}: {generated}")


main()