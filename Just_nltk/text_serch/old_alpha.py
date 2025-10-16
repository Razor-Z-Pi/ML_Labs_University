# import nltk

# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')
# nltk.download('treebank')

# sentence = """At eight o'clock on Thursday morning
# ... Arthur didn't feel very good."""

# tokens = nltk.word_tokenize(sentence)
# print(tokens)
# class TextGenerator:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Используется устройство: {self.device}")

#     def load_shakespeare_data(self):
#         """Загрузка текстов Шекспира"""
#         print("Загрузка текстов Шекспира...")
#         shakespeare_texts = []

#         # Используем тексты Шекспира из проекта Gutenberg
#         try:
#             shakespeare_texts.append(gutenberg.raw('shakespeare-hamlet.txt'))
#             shakespeare_texts.append(gutenberg.raw('shakespeare-macbeth.txt'))
#             shakespeare_texts.append(gutenberg.raw('shakespeare-caesar.txt'))
#         except:
#             print("Не удалось загрузить тексты Шекспира, используем демо-текст")
#             # Демо-текст если основные тексты недоступны
#             shakespeare_texts = [
#                 "To be or not to be that is the question",
#                 "All the world is a stage and all the men and women merely players",
#                 "What is in a name that which we call a rose by any other name would smell as sweet"
#             ]

#         text = ' '.join(shakespeare_texts)
#         return text.lower()

#     def preprocess_text(self, text, seq_length=50):
#         print("Токенизация текста...")
#         tokens = word_tokenize(text)

#         # Создаем словарь
#         self.vocab = sorted(set(tokens))
#         self.vocab_size = len(self.vocab)

#         print(f"Размер словаря: {self.vocab_size}")

#         # Создаем mapping слов в индексы и обратно
#         self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
#         self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

#         # Конвертируем текст в последовательность индексов
#         print("Создание последовательностей...")
#         encoded_text = [self.word_to_idx[word] for word in tokens]

#         # Создаем обучающие последовательности
#         sequences = []
#         targets = []

#         for i in range(0, len(encoded_text) - seq_length):
#             seq = encoded_text[i:i + seq_length]
#             target = encoded_text[i + 1:i + seq_length + 1]
#             sequences.append(seq)
#             targets.append(target)

#         return np.array(sequences), np.array(targets)

#     def create_data_loader(self, sequences, targets, batch_size=64):
#         sequences_tensor = torch.LongTensor(sequences)
#         targets_tensor = torch.LongTensor(targets)

#         dataset = torch.utils.data.TensorDataset(sequences_tensor, targets_tensor)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         return dataloader

# class ShakespeareRNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.3):
#         super(ShakespeareRNN, self).__init__()

#         self.vocab_size = vocab_size
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)

#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
#                            dropout=dropout, batch_first=True)

#         self.dropout = nn.Dropout(dropout)

#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, x, hidden=None):
#         batch_size = x.size(0)

#         x = self.embedding(x)

#         lstm_out, hidden = self.lstm(x, hidden)

#         lstm_out = self.dropout(lstm_out)

#         lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

#         output = self.fc(lstm_out)

#         return output, hidden

#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
#                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
#         return hidden

# class TextGenerationTrainer:
#     def __init__(self, model, dataloader, device):
#         self.model = model.to(device)
#         self.dataloader = dataloader
#         self.device = device

#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(model.parameters(), lr=0.001)

#         self.train_losses = []

#     def train(self, epochs=10):
#         print("Начало обучения...")
#         self.model.train()

#         for epoch in range(epochs):
#             total_loss = 0
#             hidden = None

#             for batch_idx, (data, targets) in enumerate(self.dataloader):
#                 data, targets = data.to(self.device), targets.to(self.device)

#                 self.optimizer.zero_grad()

#                 hidden = tuple([h.data for h in hidden]) if hidden is not None else None

#                 output, hidden = self.model(data, hidden)

#                 loss = self.criterion(output, targets.view(-1))

#                 loss.backward()

#                 nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)

#                 self.optimizer.step()

#                 total_loss += loss.item()

#                 if batch_idx % 100 == 0:
#                     print(f'Эпох: {epoch+1}/{epochs}, Очередь: {batch_idx}, Потерь: {loss.item():.4f}')

#             avg_loss = total_loss / len(self.dataloader)
#             self.train_losses.append(avg_loss)
#             print(f'Эпох: {epoch+1}/{epochs}, Потери: {avg_loss:.4f}')

#     def plot_training(self):
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.train_losses, label='Обучение потерь')
#         plt.xlabel('Эпох')
#         plt.ylabel('Потери')
#         plt.title('Обучение')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

# def generate_text(model, word_to_idx, idx_to_word, start_text, length=100, temperature=0.8):
#     model.eval()

#     # Конвертируем начальный текст в индексы
#     words = word_tokenize(start_text.lower())
#     encoded = [word_to_idx.get(word, 0) for word in words]

#     generated = encoded.copy()

#     with torch.no_grad():
#         hidden = None

#         for _ in range(length):
#             # Подготавливаем входные данные
#             input_seq = torch.LongTensor([encoded]).to(model.device)

#             # Получаем предсказания
#             output, hidden = model(input_seq, hidden)

#             # Применяем temperature для разнообразия
#             output = output[-1] / temperature
#             probabilities = torch.softmax(output, dim=-1)

#             # Сэмплируем следующее слово
#             predicted_idx = torch.multinomial(probabilities, 1).item()

#             # Добавляем к сгенерированной последовательности
#             generated.append(predicted_idx)
#             encoded = encoded[1:] + [predicted_idx]

#     # Конвертируем обратно в текст
#     generated_text = ' '.join([idx_to_word[idx] for idx in generated])
#     return generated_text

# def main():
#     text_gen = TextGenerator()

#     text = text_gen.load_shakespeare_data()
#     sequences, targets = text_gen.preprocess_text(text, seq_length=20)

#     dataloader = text_gen.create_data_loader(sequences, targets, batch_size=32)

#     vocab_size = text_gen.vocab_size
#     model = ShakespeareRNN(vocab_size=vocab_size,
#                           embedding_dim=128,
#                           hidden_dim=256,
#                           n_layers=2)

#     print(f"Модель создана: {model}")
#     print(f"Количество параметров: {sum(p.numel() for p in model.parameters())}")

#     # Обучение
#     trainer = TextGenerationTrainer(model, dataloader, text_gen.device)
#     trainer.train(epochs=10)

#     # Визуализация обучения
#     trainer.plot_training()

#     # Генерация текста
#     print("\n" + "="*50)
#     print("ГЕНЕРАЦИЯ ТЕКСТА")
#     print("="*50)

#     start_texts = [
#         "to be or not",
#         "all the world",
#         "what is in",
#         "shall i compare",
#         "now is the"
#     ]

#     for start_text in start_texts:
#         generated = generate_text(model, text_gen.word_to_idx, text_gen.idx_to_word,
#                                 start_text, length=50, temperature=0.7)
#         print(f"Начало: '{start_text}'")
#         print(f"Сгенерированный текст: {generated}")
#         print("-" * 80)


# main()
# tagged = nltk.pos_tag(tokens)
# print(tagged[0:6])

# entities = nltk.chunk.ne_chunk(tagged)
# print(entities)

"""
Выше пример из документациями (tutorial)
"""

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg, reuters, brown
import random
import matplotlib.pyplot as plt
from collections import Counter
import re

nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('reuters')
nltk.download('brown')

class TextGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

    def load_corpus_data(self):
        """Загрузка текстов из различных корпусов"""
        print("Загрузка текстов из корпусов...")
        all_texts = []

        # # Загрузка текстов Шекспира
        # try:
        #     shakespeare_files = ['shakespeare-hamlet.txt', 'shakespeare-macbeth.txt',
        #                        'shakespeare-caesar.txt']
        #     for file in shakespeare_files:
        #         try:
        #             text = gutenberg.raw(file)
        #             # Берем только часть текста для ускорения обучения
        #             text = text[:50000]  # Ограничиваем размер
        #             all_texts.append(text)
        #         except:
        #             continue
        # except Exception as e:
        #     print(f"Ошибка загрузки Шекспира: {e}")

        if not all_texts:
            print("Не удалось загрузить тексты, используем демо-текст")
            all_texts = [
                "заяц бежал по лесу и увидел лису. лиса сидела под деревом и смотрела на зайца. "
                "заяц быстро побежал в другую сторону. он знал что лиса может быть опасной. "
                "в лесу было много животных: белки птицы и другие зайцы. заяц бежал очень быстро "
                "чтобы спастись от лисы. холм был высоким и заяц устал бежать. но он продолжал "
                "бежать потому что боялся лисы. в конце концов заяц нашел свою нору и спрятался в ней. "
                "кот сидел на окне и смотрел на улицу. солнце светило ярко и птицы пели. "
                "дерево стояло посреди поля и качалось на ветру. ветер дул с севера и приносил прохладу. "
                "мальчик читал книгу в саду. книга была интересной и мальчик не замечал времени. "
                "девочка рисовала картину красками. картина получалась красивой и яркой. "
                "собака бежала за мячом по траве. мяч катился быстро но собака была быстрее. "
                "учитель объяснял урок ученикам. ученики слушали внимательно и задавали вопросы."
            ]

        # Объединяем все тексты
        full_text = ' '.join(all_texts)

        # Предобработка текста
        full_text = self.clean_text(full_text)

        print(f"Загружено текстов: {len(full_text)} символов, {len(full_text.split())} слов")
        return full_text.lower()

    def clean_text(self, text):
        """Очистка текста от лишних символов"""
        # Удаляем лишние пробелы и символы
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\?\!]', '', text)
        return text

    def preprocess_text(self, text, seq_length=5, min_freq=1):  # Уменьшил seq_length для стабильности
        """Улучшенная предобработка текста"""
        print("Токенизация текста...")

        # Токенизация с учетом пунктуации
        tokens = word_tokenize(text)

        print(f"Всего токенов до фильтрации: {len(tokens)}")

        # Фильтрация редких слов
        word_freq = Counter(tokens)
        filtered_tokens = [word for word in tokens if word_freq[word] >= min_freq]

        print(f"Токенов после фильтрации: {len(filtered_tokens)}")

        # Создаем словарь
        self.vocab = sorted(set(filtered_tokens))
        self.vocab_size = len(self.vocab)

        print(f"Размер словаря: {self.vocab_size}")
        print(f"Примеры слов в словаре: {list(self.vocab)[:10]}")

        # Создаем mapping слов в индексы и обратно
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

        # Конвертируем текст в последовательность индексов
        print("Создание последовательностей...")
        encoded_text = [self.word_to_idx[word] for word in filtered_tokens]

        # Создаем обучающие последовательности
        sequences = []
        targets = []

        for i in range(0, len(encoded_text) - seq_length):
            seq = encoded_text[i:i + seq_length]
            target = encoded_text[i + seq_length]  # Только следующее слово
            sequences.append(seq)
            targets.append(target)

        print(f"Создано последовательностей: {len(sequences)}")
        return np.array(sequences), np.array(targets)

class ImprovedShakespeareRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.2):
        super(ImprovedShakespeareRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        # Выходной слой
        self.fc = nn.Linear(hidden_dim, vocab_size)

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

        # Выходной слой
        output = self.fc(lstm_out)

        return output, hidden

class TextGenerationTrainer:
    def __init__(self, model, dataloader, device, word_to_idx, idx_to_word):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        self.train_losses = []

    def train(self, epochs=10):
        print("Начало обучения...")
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0

            for batch_idx, (data, targets) in enumerate(self.dataloader):
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                # Прямой проход
                output, _ = self.model(data)

                # Вычисление потерь
                loss = self.criterion(output, targets)

                # Обратный проход
                loss.backward()

                # Обновление весов
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                if batch_idx % 50 == 0:
                    print(f'Эпоха: { epoch + 1 }/{ epochs }, Батч: {batch_idx}, Потери: {loss.item():.4f}')

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                self.train_losses.append(avg_loss)
                print(f'Эпоха: { epoch + 1 } / { epochs }, Средние потери: {avg_loss:.4f}')

            # Генерация примера текста после каждой эпохи
            if (epoch + 1) % 2 == 0:
                self.generate_example(epoch + 1)

    def generate_example(self, epoch):
        """Генерация примера текста во время обучения"""
        start_text = "заяц бежал"
        generated = generate_text(self.model, self.word_to_idx, self.idx_to_word,
                                start_text, length = 10, temperature = 0.8)
        print(f"Пример генерации (эпоха {epoch}): {generated}")

    def plot_training(self):
        plt.figure(figsize = (10, 6))
        plt.plot(self.train_losses, label='Потери при обучении')
        plt.xlabel('Эпохи')
        plt.ylabel('Потери')
        plt.title('Процесс обучения модели')
        plt.legend()
        plt.grid(True)
        plt.show()

def generate_text(model, word_to_idx, idx_to_word, start_text, length = 15, temperature = 0.8):
    model.eval()

    # Токенизация начального текста
    words = word_tokenize(start_text.lower())

    # Конвертируем в индексы
    encoded = []
    for word in words:
        if word in word_to_idx:
            encoded.append(word_to_idx[word])
        else:
            # Если слово неизвестно, пропускаем
            continue

    if not encoded:
        # Если все слова неизвестны, используем случайное слово
        encoded = [random.randint(0, len(word_to_idx) - 1)]

    current_sequence = encoded.copy()
    generated_indices = encoded.copy()

    with torch.no_grad():
        hidden = None

        for _ in range(length):
            # Подготавливаем входные данные
            if len(current_sequence) < 5:  # Дополняем до нужной длины
                input_seq = current_sequence + [0] * (5 - len(current_sequence))
            else:
                input_seq = current_sequence[-5:]  # Берем последние 5 слов

            input_tensor = torch.LongTensor([input_seq]).to(model.device)

            # Получаем предсказания
            output, hidden = model(input_tensor, hidden)

            # Применяем temperature
            output = output / temperature
            probabilities = torch.softmax(output, dim = -1)

            # Сэмплируем следующее слово
            predicted_idx = torch.multinomial(probabilities, 1).item()

            # Добавляем к сгенерированной последовательности
            generated_indices.append(predicted_idx)

            # Обновляем текущую последовательность
            current_sequence.append(predicted_idx)
            if len(current_sequence) > 10:  # Ограничиваем длину
                current_sequence = current_sequence[-5:]

    # Конвертируем обратно в текст
    generated_words = []
    for idx in generated_indices:
        if idx in idx_to_word:
            generated_words.append(idx_to_word[idx])

    generated_text = ' '.join(generated_words)
    return generated_text

def main():
    # Инициализация
    text_gen = TextGenerator()

    # Загрузка и предобработка данных
    text = text_gen.load_corpus_data()
    sequences, targets = text_gen.preprocess_text(text, seq_length = 5, min_freq = 1)

    # Проверяем данные
    print(f"Размер последовательности: {sequences.shape}")
    print(f"Размер таргеты: {targets.shape}")

    if len(sequences) == 0:
        print("Ошибка: не создано обучающих последовательностей!!!")
        return

    # Создание DataLoader
    sequences_tensor = torch.LongTensor(sequences)
    targets_tensor = torch.LongTensor(targets)

    dataset = torch.utils.data.TensorDataset(sequences_tensor, targets_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)

    # Создание модели
    vocab_size = text_gen.vocab_size
    model = ImprovedShakespeareRNN(vocab_size=vocab_size,
                                 embedding_dim = 128,
                                 hidden_dim = 256,
                                 n_layers = 2)

    model.device = text_gen.device

    print(f"Модель создана")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Обучение
    trainer = TextGenerationTrainer(model, dataloader, text_gen.device,
                                  text_gen.word_to_idx, text_gen.idx_to_word)
    trainer.train(epochs = 30)

    # Визуализация обучения
    trainer.plot_training()

    # Демонстрация генерации
    print("\n" + "=" * 50)
    print("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ТЕКСТА!!!")
    print("=" * 50)

    test_phrases = [
        "заяц бежал",
        "кот сидел",
        "солнце светило",
        "дерево стояло",
        "ветер дул"
    ]

    for phrase in test_phrases:
        generated = generate_text(model, text_gen.word_to_idx, text_gen.idx_to_word,
                                phrase, length = 12, temperature = 0.7)
        print(f"Начало: '{phrase}'")
        print(f"Продолжение: {generated}")
        print("-" * 80)
main()