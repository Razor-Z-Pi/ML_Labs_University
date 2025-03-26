import numpy as np

# S(x) = 1 / 1 + e^-x
# Сигмоидная функция
# Входной слой 25 нейронов 5х5 матрица
# скрытый слой сигмоидная активация 10 неронов
# выходной слой 10 нейронов софтмакс (на каждую цифру отдельно)
#

# Использовал со старой версии 
def sigmoid(x): # диапозон от (0, 1)
    return 1 / (1 + np.exp(-x))

# Производная сигмоиды
def sigmoid_derivative(x): # Используеться только раз при вычесление скрытого слоя
    return x * (1 - x)

# Функция softmax (для вироятности)
# Улучшенная версия, изначально я сделал перебор массивам 
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

class Neron:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        
    def forward(self, X):
        # Прямое распространение
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = softmax(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output
    
    def train(self, X, y, epochs=1000, lr=0.1):
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Прямое распространение (предсказание я бы назвал)
            output = self.forward(X)
            
            # Вычисление ошибки
            error = output - y
            loss = -np.sum(y * np.log(output + 1e-10)) / len(X)
            losses.append(loss)
            
            # Обратное распространение
            d_output = error
            d_weights2 = np.dot(self.hidden.T, d_output)
            d_bias2 = np.sum(d_output, axis=0, keepdims=True)
            
            d_hidden = np.dot(d_output, self.weights2.T) * sigmoid_derivative(self.hidden)
            d_weights1 = np.dot(X.T, d_hidden)
            d_bias1 = np.sum(d_hidden, axis=0, keepdims=True)
            
            # Обновление весов
            self.weights1 -= lr * d_weights1
            self.weights2 -= lr * d_weights2
            self.bias1 -= lr * d_bias1
            self.bias2 -= lr * d_bias2
            
            # Точность
            predictions = np.argmax(output, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
            accuracies.append(accuracy)
            
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, Потери: {loss:.4f}, Точность: {accuracy:.4f}")
        
        return losses, accuracies

# обучающие данные (матрицы 5x5 для цифр 0-9)
def create_data_num():
    digits = [
        # 0
        [[1,1,1,1,1],
         [1,0,0,0,1],
         [1,0,0,0,1],
         [1,0,0,0,1],
         [1,1,1,1,1]],
        # 1
        [[0,0,1,0,0],
         [0,1,1,0,0],
         [0,0,1,0,0],
         [0,0,1,0,0],
         [0,1,1,1,0]],
        # 2
        [[1,1,1,1,1],
         [0,0,0,0,1],
         [1,1,1,1,1],
         [1,0,0,0,0],
         [1,1,1,1,1]],
        # 3
        [[1,1,1,1,1],
         [0,0,0,0,1],
         [0,1,1,1,1],
         [0,0,0,0,1],
         [1,1,1,1,1]],
        # 4
        [[1,0,0,0,1],
         [1,0,0,0,1],
         [1,1,1,1,1],
         [0,0,0,0,1],
         [0,0,0,0,1]],
        # 5
        [[1,1,1,1,1],
         [1,0,0,0,0],
         [1,1,1,1,1],
         [0,0,0,0,1],
         [1,1,1,1,1]],
        # 6
        [[1,1,1,1,1],
         [1,0,0,0,0],
         [1,1,1,1,1],
         [1,0,0,0,1],
         [1,1,1,1,1]],
        # 7
        [[1,1,1,1,1],
         [0,0,0,0,1],
         [0,0,0,1,0],
         [0,0,1,0,0],
         [0,1,0,0,0]],
        # 8
        [[1,1,1,1,1],
         [1,0,0,0,1],
         [1,1,1,1,1],
         [1,0,0,0,1],
         [1,1,1,1,1]],
        # 9
        [[1,1,1,1,1],
         [1,0,0,0,1],
         [1,1,1,1,1],
         [0,0,0,0,1],
         [1,1,1,1,1]]
    ]
    
    X = np.array(digits).reshape(10, 25)  # 10 цифр, каждая 5x5=25 пикселей
    y = np.eye(10)
    return X, y

# Создаем и обучаем сеть
X, y = create_data_num()
network = Neron(input_size=25, hidden_size=10, output_size=10)
losses, accuracies = network.train(X, y, epochs=2000, lr=0.1)

# Тестирование
def test_num(matrix):
    flat = np.array(matrix).flatten().reshape(1, -1)
    output = network.forward(flat)
    number = np.argmax(output)
    confidence = output[0, number]
    
    print(f"\nМатрица:")
    for row in matrix:
        print(" ".join(str(x) for x in row))
    
    print(f"\nСеть считает, что это: {number} (уверенность: {confidence:.4f})")
    print("Выходные вероятности:")
    for i, prob in enumerate(output[0]):
        print(f"{i}: {prob:.4f}")

# Тестовые примеры
test_0 = [[1,1,1,1,1],
          [1,0,0,0,1],
          [1,0,0,0,1],
          [1,0,0,0,1],
          [1,1,1,1,1]]

test_5 = [[1,1,1,1,1],
          [1,0,0,0,0],
          [1,1,1,1,1],
          [0,0,0,0,1],
          [1,1,1,1,1]]

test_noise = [[1,1,0,1,1],
              [1,0,0,0,1],
              [1,1,1,1,1],
              [0,0,0,0,1],
              [1,1,1,1,1]]  # Похоже на 9 с шумом

test_num(test_0)
test_num(test_5)
test_num(test_noise)