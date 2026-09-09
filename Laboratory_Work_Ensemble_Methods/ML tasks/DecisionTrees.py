import numpy as np

def gini(groups, classes):
    """Вычисление Gini impurity для сплита"""
    n_instances = float(sum([len(group) for group in groups]))
    gini_score = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0: continue
        score = 0.0
        # Для каждого класса в группе
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini_score += (1.0 - score) * (size / n_instances)
    return gini_score

def test_split(index, value, dataset):
    """Разделение данных на лево и право"""
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_best_split(dataset):
    """Перебор всех признаков и порогов для поиска лучшего сплита"""
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini_score = gini(groups, class_values)
            if gini_score < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini_score, groups
    return {'Индекс':b_index, 'Значение':b_value, 'Группы':b_groups}

def to_terminal(group):
    """Возвращает самый частый класс в листе"""
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key = outcomes.count)

def split(node, max_depth, min_size, depth):
    """Рекурсивное построение дерева"""
    left, right = node['Группы']
    del(node['Группы'])
    # Проверка на лист
    if not left or not right:
        node['Лево'] = node['Право'] = to_terminal(left + right)
        return
    # Проверка максимальной глубины
    if depth >= max_depth:
        node['Лево'], node['Право'] = to_terminal(left), to_terminal(right)
        return
    # Левая ветка
    if len(left) <= min_size:
        node['Лево'] = to_terminal(left)
    else:
        node['Лево'] = get_best_split(left)
        split(node['Лево'], max_depth, min_size, depth+1)
    # Правая ветка
    if len(right) <= min_size:
        node['Право'] = to_terminal(right)
    else:
        node['Право'] = get_best_split(right)
        split(node['Право'], max_depth, min_size, depth+1)

dataset = [[50, 5, 1], [60, 6, 1], [30, 4, 1], [80, 10, 0], [20, 3, 1], [90, 12, 0]]
tree = get_best_split(dataset)
split(tree, max_depth = 2, min_size = 1, depth = 1)
print("Построенное дерево:", tree)