import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Константы для среды
ACTIONS = ['вверх', 'вниз', 'влево', 'вправо']
ACTION_EFFECTS = {
    'вверх': (-1, 0),
    'вниз': (1, 0),
    'влево': (0, -1),
    'вправо': (0, 1)
}

# Константы для наград
REWARDS = {
    'goal': 10,
    'obstacle': -5,
    'step': -0.1,
    'default': 0
}

def create_environment(height = 5, width = 5):
    env = {
        'height': height,
        'width': width,
        'start': (0, 0),
        'goal': (height-1, width-1),
        'obstacles': [(1, 1), (2, 2), (3, 1)],
        'current_state': (0, 0)
    }
    return env

def reset_environment(env): # сброс среды до начального состояния
    env['current_state'] = env['start']
    return env['current_state']

def get_all_states(env): # собираем все стостояние
    return [(i, j) for i in range(env['height']) for j in range(env['width'])]

def is_valid_state(env, state): # Проверка валидности состояния
    i, j = state
    return (0 <= i < env['height'] and 
            0 <= j < env['width'] and 
            state not in env['obstacles'])

def step_from_state(env, state, action): # Выполнение действия из указанного состояния
    i, j = state
    di, dj = ACTION_EFFECTS[action]
    new_state = (i + di, j + dj)
    
    # Проверка валидности
    if not is_valid_state(env, new_state):
        return state, REWARDS['obstacle'], False
    
    # Проверка на препятствия
    if new_state in env['obstacles']:
        return state, REWARDS['obstacle'], False
    
    # Проверка на цель
    if new_state == env['goal']:
        return new_state, REWARDS['goal'], True
    
    return new_state, REWARDS['step'], False

def step(env, action): # Выполнение действия из текущего состояния
    next_state, reward, done = step_from_state(env, env['current_state'], action)
    env['current_state'] = next_state
    return next_state, reward, done

def initialize_q_table(env):
    states = get_all_states(env)
    q_table = {}
    
    for state in states:
        q_table[state] = {}
        for action in ACTIONS:
            # Особая инициализация для препятствий и цели
            if state in env['obstacles']:
                q_table[state][action] = -1000.0
            elif state == env['goal']:
                q_table[state][action] = 100.0
            else:
                q_table[state][action] = 0.0
    
    return q_table

def get_action(q_table, state, epsilon, training = True): # Выбор действия с использованием жадной стратегии
    if training and random.random() < epsilon:
        # Исследование: случайное действие
        return random.choice(ACTIONS)
    else:
        # Лучшее действие из Q-таблицы
        q_values = q_table[state]
        # Находим действие с максимальным Q-значением
        best_actions = []
        best_value = float('-inf')
        
        for action, value in q_values.items():
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        
        if best_actions:
            return random.choice(best_actions)
        else:
            return random.choice(ACTIONS)

def update_q_value(q_table, state, action, reward, next_state, lr, gamma): #  Обновление Q-значения по уравнению Беллмана
    # Текущее Q-значение
    current_q = q_table[state][action]
    
    # Максимальное Q-значение для следующего состояния
    next_q_values = q_table[next_state]
    max_next_q = max(next_q_values.values())
    
    # Обновление по формуле Беллмана
    new_q = current_q + lr * (reward + gamma * max_next_q - current_q)
    
    # Сохраняем обновленное значение
    q_table[state][action] = new_q

def train_agent(env, q_table, episodes = 500, max_steps = 100, 
                lr = 0.1, gamma = 0.95, epsilon = 0.3, 
                epsilon_decay = 0.995, min_epsilon = 0.01, 
                verbose = True, print_freq = 50):
    rewards_history = []
    steps_history = []
    current_epsilon = epsilon
    
    for episode in range(episodes):
        state = reset_environment(env)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Выбор действия
            action = get_action(q_table, state, current_epsilon, training = True)
            
            # Выполнение действия
            next_state, reward, done = step(env, action)
            
            # Обновление Q-значения
            update_q_value(q_table, state, action, reward, next_state, lr, gamma)
            
            # Переход в новое состояние
            state = next_state
            total_reward += reward
            steps += 1
        
        # Сохраняем метрики
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # Уменьшаем epsilon
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        
        # Вывод прогресса
        if verbose and (episode + 1) % print_freq == 0:
            avg_reward = np.mean(rewards_history[-print_freq:])
            avg_steps = np.mean(steps_history[-print_freq:])
            print(f"Эпизод {episode + 1}: "
                  f"Средняя награда = {avg_reward:.2f}, "
                  f"Средние шаги = {avg_steps:.2f}, "
                  f"Epsilon = {current_epsilon:.3f}")
    
    return rewards_history, steps_history

def get_optimal_policy(q_table, env):
    policy = {}
    states = get_all_states(env)
    
    for state in states:
        if state == env['goal']:
            policy[state] = "ЦЕЛЬ"
        elif state in env['obstacles']:
            policy[state] = "ПРЕПЯТСТВИЕ"
        else:
            q_values = q_table[state]
            best_action = max(q_values, key=q_values.get)
            policy[state] = best_action
    
    return policy

def find_path_to_goal(q_table, env, start_state = None):
    if start_state is None:
        start_state = env['start']
    
    path = [start_state]
    current_state = start_state
    visited = set([start_state])
    max_steps = 100
    
    for _ in range(max_steps):
        if current_state == env['goal']:
            break
        
        # Получаем лучшее действие
        q_values = q_table[current_state]
        best_action = max(q_values, key=q_values.get)
        
        # Выполняем действие
        next_state, _, _ = step_from_state(env, current_state, best_action)
        
        # Проверка на зацикливание
        if next_state in visited:
            # Если зациклились, выбираем другое действие
            available_actions = []
            for action in ACTIONS:
                ns, _, _ = step_from_state(env, current_state, action)
                if ns not in visited and ns != current_state:
                    available_actions.append(action)
            
            if available_actions:
                next_state, _, _ = step_from_state(
                    env, current_state, random.choice(available_actions))
            else:
                break
        
        path.append(next_state)
        visited.add(next_state)
        current_state = next_state
        
        if current_state == env['goal']:
            break
    
    return path

def render_environment(env, path = None): # Визуализация среды
    grid = [['.' for _ in range(env['width'])] for _ in range(env['height'])]
    
    # Размещаем препятствия
    for obs in env['obstacles']:
        i, j = obs
        grid[i][j] = '|'
    
    # Размещаем цель
    gi, gj = env['goal']
    grid[gi][gj] = 'x'
    
    # Размещаем старт
    si, sj = env['start']
    grid[si][sj] = 'S'
    
    # Размещаем путь, если есть
    if path:
        for i, (pi, pj) in enumerate(path[1:-1]):  # Пропускаем старт и цель
            grid[pi][pj] = 'o'
    
    print("\nСостояние среды:")
    print("*" * 30)
    for row in grid:
        print(' '.join(row))

def visualize_policy(q_table, env): # Визуализация политики
    policy = get_optimal_policy(q_table, env)
    
    action_symbols = {
        'вверх': 'вверх',
        'вниз': 'вниз',
        'влево': 'влево',
        'вправо': 'вправо',
        'Цель': 'x',
        'Препятствие': '|'
    }
    
    print("\nОпт. палитика:")
    print("*" * 30)
    for i in range(env['height']):
        row = []
        for j in range(env['width']):
            state = (i, j)
            action = policy.get(state, '?')
            symbol = action_symbols.get(action, '?')
            row.append(symbol)
        print(' '.join(row))
    
    # Находим и выводим путь
    path = find_path_to_goal(q_table, env)
    print(f"\nНайденный путь к цели:")
    path_str = ' => '.join([f"({s[0]},{s[1]})" for s in path])
    print(path_str)
    print(f"Длина пути: {len(path)} шагов")
    
    return path

def plot_training_results(rewards_history, steps_history):
    plt.figure(figsize = (12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Награды по эпизодам')
    plt.xlabel('Эпизод')
    plt.ylabel('Суммарная награда')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(steps_history)
    plt.title('Шаги по эпизодам')
    plt.xlabel('Эпизод')
    plt.ylabel('Количество шагов')
    plt.grid(True)
    
    plt.tight_layout()

    plt.show()

def print_q_values_for_states(q_table, env, states_list):   
    for state in states_list:
        if state in q_table:
            print(f"\nСостояние {state}:")
            for action in ACTIONS:
                q_value = q_table[state][action]
                print(f"  {action}: {q_value:.3f}")

env = create_environment(5, 5)

# Показываем начальное состояние среды
render_environment(env)

# Инициализируем Q-таблицу
q_table = initialize_q_table(env)

# Обучаем агента
print("\nОбучаем...")
print("*" * 40)
rewards_history, steps_history = train_agent(
    env, q_table,
    episodes = 500,
    lr = 0.1,
    gamma = 0.95,
    epsilon = 0.3,
    verbose = True,
    print_freq = 50
)

# Визуализируем результаты
print("\n" + "*" * 70)
print("Результаты обучения")
print("*" * 70)

# Показываем оптимальную политику и путь
path = visualize_policy(q_table, env)

# Показываем среду с найденным путем
render_environment(env, path)

# Выводим Q-значения для ключевых состояний
key_states = [(0, 0), (2, 0), (4, 4)]
print_q_values_for_states(q_table, env, key_states)

# Строим графики обучения
plot_training_results(rewards_history, steps_history)

test_starts = [(0, 2), (1, 0), (3, 3)]
for start in test_starts:
        if start not in env['obstacles'] and start != env['goal']:
            path = find_path_to_goal(q_table, env, start)
            print(f"\nСтарт {start} => Цель: {' => '.join([f'({s[0]},{s[1]})' for s in path])}")