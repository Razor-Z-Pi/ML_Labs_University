import random

class Neuron:
    def __init__(self, input_size):
        # Инициализация весов и смещения случайными значениями
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)

    def sigmoid(self, x):
        # Функция активации (сигмоида)
        return 1 / (1 + 2.71828 ** -x)

    def sigmoid_derivative(self, x):
        # Производная сигмоиды
        return x * (1 - x)

    def feedforward(self, inputs):
        # Вычисление выхода нейрона
        total = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.sigmoid(total)

    def train(self, inputs, target, learning_rate=0.1):
        # Обучение нейрона
        output = self.feedforward(inputs)
        error = target - output  # Ошибка
        gradient = error * self.sigmoid_derivative(output)  # Градиент

        # Обновление весов и смещения
        self.weights = [w + learning_rate * gradient * i for w, i in zip(self.weights, inputs)]
        self.bias += learning_rate * gradient


training_data = [
    ([1, 1, 1, 
      1, 0, 0, 
      1, 1, 1, 
      0, 0, 1, 
      1, 1, 1], 1), 

    ([1, 0, 1, 
      1, 1, 0, 
      1, 1, 1, 
      0, 0, 0, 
      1, 1, 1], 0), 
]

neuron = Neuron(input_size=15)

for _ in range(10000):
    for inputs, target in training_data:
        neuron.train(inputs, target)

# Тестирование нейрона
test_data = [
    ([  1, 1, 1, 
        1, 0, 0, 
        1, 1, 1, 
        0, 0, 1, 
        1, 1, 1], 
        "Да, это число 5"),

    ([  1, 0, 1, 
        1, 1, 0,
        1, 1, 1, 
        0, 0, 0, 
        1, 1, 1], 
        "Нет, это не число 5"),
]

for inputs, expected in test_data:
    output = neuron.feedforward(inputs)
    print(f"Вход: {inputs}, Выход: {output:.2f}, Ожидаемый результат: {expected}")