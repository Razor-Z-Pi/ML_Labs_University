import numpy as np
import random

grid_size = 5
goal_state = (4, 4)
num_episodes = 1000
alpha = 0.1 # скорость обучения
gamma = 0.9 # коэффициент дисконтирования
epsilon = 0.1 # вероятность случайного действия

q_table = np.zeros((grid_size, grid_size, 4)) # 4 действия: вверх, вниз, влево, вправо

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3) # случайное действие
    else:
        return np.argmax(q_table[state]) # лучшее действие


def take_action(state, action):
    x, y = state
    if action == 0 and x > 0: # вверх
        return (x- 1, y)
    elif action == 1 and x < grid_size - 1: # вниз
        return (x + 1, y)
    elif action == 2 and y > 0: # влево
        return (x, y - 1)
    elif action == 3 and y < grid_size - 1: # вправо
        return (x, y + 1)
    return state # если действие недопустимо, остаемся на месте

for episode in range (num_episodes):
    state = (0, 0)
    while state != goal_state:
        action = choose_action(state)
        next_state = take_action(state, action)

        # Награда: 1 за достижение цели, 0 в противном случае
        reward = 1 if next_state == goal_state else 0

        # Обновление Q-таблицы
        q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

        state = next_state

# Проверка, как агент научился двигаться к цели
state = (0, 0)
path = [state]
while state != goal_state:
    action = np.argmax(q_table[state])
    state = take_action(state, action)
    path.append(state)

print ("Найденный путь к цели:", path)