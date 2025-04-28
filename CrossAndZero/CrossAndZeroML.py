import numpy as np
import random
import tensorflow as tf
from collections import deque
import os

print("*" * 10, " Крестики-нолики с ИИ на TensorFlow ", "*" * 10)

# Параметры обучения
EPISODES = 1000  # Количество игр для обучения
MEMORY_SIZE = 1000  # Размер памяти для опыта
BATCH_SIZE = 32  # Размер батча для обучения
GAMMA = 0.95  # Коэффициент дисконтирования
EPSILON = 1.0  # Начальная вероятность случайного хода
EPSILON_MIN = 0.01  # Минимальная вероятность случайного хода
EPSILON_DECAY = 0.995  # Скорость уменьшения epsilon

# Создаем модель нейронной сети
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(9, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

# Класс для агента DQN
class DQNAgent:
    def __init__(self):
        self.model = create_model()
        self.target_model = create_model()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        
        # Загружаем модель, если она существует
        if os.path.exists('tic_tac_toe_model.h5'):
            self.model.load_weights('tic_tac_toe_model.h5')
            self.target_model.load_weights('tic_tac_toe_model.h5')
            self.epsilon = EPSILON_MIN

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(available_moves)
        
        state = np.reshape(state, [1, 9])
        act_values = self.model.predict(state, verbose=0)
        
        # Фильтруем только доступные ходы
        available_actions = act_values[0][available_moves]
        return available_moves[np.argmax(available_actions)]

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.amax(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def save_model(self):
        self.model.save_weights('tic_tac_toe_model.h5')

# Инициализация агента
agent = DQNAgent()

# Функции игры
def draw_board(board):
    print("-" * 13)
    for i in range(3):
        print("|", board[0 + i * 3], "|", board[1 + i * 3], "|", board[2 + i * 3], "|")
        print("-" * 13)

def get_state(board):
    """Преобразует доску в числовой формат для нейросети"""
    state = np.zeros(9)
    for i in range(9):
        if board[i] == 'X':
            state[i] = 1
        elif board[i] == 'O':
            state[i] = -1
    return state

def get_available_moves(board):
    """Возвращает список доступных ходов"""
    return [i for i in range(9) if board[i] not in ['X', 'O']]

def check_win(board):
    win_coord = ((0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6))
    for i in win_coord:
        if board[i[0]] == board[i[1]] == board[i[2]]:
            return board[i[0]]
    return None

def play_game(train_mode=True):
    board = [str(i+1) for i in range(9)]
    done = False
    winner = None
    
    # Определяем, кто ходит первым (случайно)
    player_turn = random.choice(['X', 'O'])
    
    while not done:
        state = get_state(board)
        available_moves = get_available_moves(board)
        
        if not available_moves:
            done = True
            break
            
        if player_turn == 'X':
            # Ход человека
            draw_board(board)
            valid = False
            while not valid:
                move = input("Куда поставим X? (1-9) ")
                try:
                    move = int(move) - 1
                    if move in available_moves:
                        valid = True
                    else:
                        print("Недопустимый ход!")
                except:
                    print("Введите число от 1 до 9!")
        else:
            # Ход ИИ
            move = agent.act(state, available_moves)
            print(f"ИИ выбирает позицию {move + 1}")
        
        # Делаем ход
        board[move] = player_turn
        
        # Проверяем победу
        winner = check_win(board)
        if winner:
            done = True
            draw_board(board)
            print(f"{winner} победил!")
            
            if train_mode and player_turn == 'O':
                # ИИ выиграл
                reward = 1
                agent.remember(state, move, reward, get_state(board), done)
            elif train_mode and player_turn == 'X':
                # ИИ проиграл
                reward = -1
                agent.remember(state, move, reward, get_state(board), done)
        elif len(get_available_moves(board)) == 0:
            # Ничья
            done = True
            draw_board(board)
            print("Ничья!")
            
            if train_mode:
                reward = 0.1
                agent.remember(state, move, reward, get_state(board), done)
        elif train_mode and player_turn == 'O':
            # Промежуточный ход ИИ
            reward = 0
            agent.remember(state, move, reward, get_state(board), done)
        
        # Меняем игрока
        player_turn = 'O' if player_turn == 'X' else 'X'
    
    if train_mode:
        agent.replay()
        agent.update_target_model()

# Обучение ИИ
print("Обучение ИИ...")
for e in range(EPISODES):
    play_game(train_mode=True)
    if e % 100 == 0:
        print(f"Эпизод: {e}, Epsilon: {agent.epsilon:.2f}")
        agent.save_model()

# Игра с обученным ИИ
print("\nОбучение завершено! Давайте сыграем!")
agent.epsilon = 0  # Отключаем случайные ходы
while True:
    play_game(train_mode=False)
    again = input("Хотите сыграть еще? (y/n): ")
    if again.lower() != 'y':
        break

agent.save_model()
print("Модель сохранена в 'tic_tac_toe_model.h5'")