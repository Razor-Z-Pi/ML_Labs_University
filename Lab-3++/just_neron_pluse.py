import random

# Обучающая выборка (идеальное изображение цифр от 0 до 9)
num0 = list('111101101101111')
num1 = list('001001001001001')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')

# Список всех цифр от 0 до 9 в едином массиве
nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]

n_sensor = 15  # количество сенсоров
weights = [[0 for _ in range(n_sensor)] for _ in range(10)]  # Веса для каждой цифры от 0 до 9

# Функция определения - является ли полученное изображение числом num
# возвращает Да, если признано, что это num. Нет, если отвергнуто, что это num
def perceptron(Sensor, num):
    b = 7  # Порог функции активации
    s = 0  # Начальное значение суммы
    for i in range(n_sensor):  # цикл суммирования сигналов от сенсоров
        s += int(Sensor[i]) * weights[num][i]
    if s >= b:
        return True  # Сумма превысила порог
    else:
        return False  # Сумма меньше порога

# Уменьшение значений весов
# Если сеть ошиблась и выдала Да при входной цифре, отличной от num
def decrease(number, num):
    for i in range(n_sensor):
        if int(number[i]) == 1:  # Возбужденный ли вход
            weights[num][i] -= 1  # Уменьшаем связанный с входом вес на единицу

# Увеличение значений весов
# Если сеть не ошиблась и выдала Да при поданной на вход цифре num
def increase(number, num):
    for i in range(n_sensor):
        if int(number[i]) == 1:  # Возбужденный ли вход
            weights[num][i] += 1  # Увеличиваем связанный с входом вес на единицу

# Тренировка сети
n = 100000  # количество уроков
for i in range(n):
    j = random.randint(0, 9)  # Генерируем случайное число j от 0 до 9
    r = perceptron(nums[j], j)  # Результат обращения к сумматору (ответ - Да или НЕТ)

    if r:  # Если сумматор сказал ДА (это цифра j)
        for k in range(10):
            if k != j:
                decrease(nums[j], k)  # наказываем сеть (уменьшаем значения весов для других цифр)
    else:  # Если сумматор сказал НЕТ (это не цифра j)
        increase(nums[j], j)  # поощряем сеть (увеличиваем значения весов для цифры j)

print(weights)  # Вывод значений весов

# Функция для определения, какая цифра изображена на входе
def recognize_number(Sensor):
    max_sum = -1  # Максимальная сумма
    recognized_num = -1  # Распознанная цифра

    for num in range(10):  # Перебираем все цифры от 0 до 9
        s = 0  # Начальное значение суммы
        for i in range(n_sensor):  # Суммируем взвешенные входы
            s += int(Sensor[i]) * weights[num][i]
        
        if s > max_sum:  # Если текущая сумма больше максимальной
            max_sum = s
            recognized_num = num

    return recognized_num  # Возвращаем распознанную цифру

# Проверка работы программы на обучающей выборке
print("+++++++++++++")
print("Проверка на обучающей выборке:")
for i in range(10):
    recognized = recognize_number(nums[i])
    print(f"На входе {i}, распознано: {recognized}")

# Тестовая выборка (различные варианты изображения цифры 5)
num51 = list('111100111000111')
num52 = list('111100010001111')
num53 = list('111100011001111')
num54 = list('110100111001111')
num55 = list('110100111001011')
num56 = list('111100101001111')

# Тестовая выборка для других цифр (примеры)
num01 = list('111101101100111')  # Немного измененный 0
num12 = list('001001001000001')  # Немного измененный 1
num23 = list('111000111100111')  # Немного измененный 2

print("+++++++++++++")
print("Проверка на тестовой выборке:")
print("Узнал 5 в 51? ", recognize_number(num51))
print("Узнал 5 в 52? ", recognize_number(num52))
print("Узнал 5 в 53? ", recognize_number(num53))
print("Узнал 5 в 54? ", recognize_number(num54))
print("Узнал 5 в 55? ", recognize_number(num55))
print("Узнал 5 в 56? ", recognize_number(num56))
print("Узнал 0 в 01? ", recognize_number(num01))
print("Узнал 1 в 12? ", recognize_number(num12))
print("Узнал 2 в 23? ", recognize_number(num23))