import numpy as np
import pandas as pd

def create_grid_3x3_q_table():
    # Определяем состояния (9 клеток)
    states = [(i, j) for i in range(3) for j in range(3)]
    
    # Определяем возможные действия
    actions = ['вверх', 'вниз', 'влево', 'вправо']
    
    # Инициализируем Q-таблицу нулями
    q_table = np.zeros((len(states), len(actions)))
    
    q_df = pd.DataFrame(
        q_table,
        index = [f"({i},{j})" for i, j in states],
        columns = actions
    )
    
    print("Q-таблица для сетки 3x3:")
    print("*" * 50)
    print(q_df)
    print("\nФормат: строки - состояния (координаты), столбцы - действия")
    
    return q_table, states, actions

q_table, states, actions = create_grid_3x3_q_table()