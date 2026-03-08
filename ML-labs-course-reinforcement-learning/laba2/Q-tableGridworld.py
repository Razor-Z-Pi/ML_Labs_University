import numpy as np
import pandas as pd

def create_gridworld_q_table(height = 4, width = 4):
    # Состояния
    states = [(i, j) for i in range(height) for j in range(width)]
    actions = ['вверх', 'вниз', 'влево', 'вправо']
    
    # Особые состояния
    start_state = (0, 0)
    goal_state = (height - 1, width - 1)
    obstacles = [(1, 1), (2, 2)]  # Препятствия
    
    # Q-таблица с учетом особых состояний
    q_table = np.zeros((len(states), len(actions)))
    
    # Для препятствий - очень низкие значения
    for obstacle in obstacles:
        if obstacle in states:
            state_idx = states.index(obstacle)
            q_table[state_idx, :] = -1000
    
    # Для целевого состояния - высокие значения
    goal_idx = states.index(goal_state)
    q_table[goal_idx, :] = 100
    
    q_df = pd.DataFrame(
        q_table,
        index=[f"({i},{j})" for i, j in states],
        columns = actions
    )
    
    print("*" * 70)
    print(f"Q-таблица для GRIDWORLD {height} x {width}")
    print("*" * 70)
    print(q_df)
    print("\nОсобые состояния:")
    print(f"  Старт: {start_state}")
    print(f"  Цель: {goal_state} (Q-значения = 100)")
    print(f"  Препятствия: {obstacles} (Q-значения = -1000)")
    
    return {
        'q_table': q_table,
        'states': states,
        'actions': actions,
        'start': start_state,
        'goal': goal_state,
        'obstacles': obstacles
    }

gridworld_data = create_gridworld_q_table(4, 4)