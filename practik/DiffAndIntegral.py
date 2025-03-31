import numpy as np

# Исходные данные
x = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
y = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

# 1. Численные производные
print("Численные производные первого порядка:")
for i in range(len(x)-1):
    dy = (y[i+1] - y[i]) / (x[i+1] - x[i])
    print(f"Интервал [{x[i]}, {x[i+1]}]: y' = {dy:.2f}")

print("\nЧисленные производные второго порядка:")
x_mid = (x[:-1] + x[1:]) / 2
dy = np.diff(y) / np.diff(x)
for i in range(len(dy) - 1):
    d2y = (dy[i+1] - dy[i]) / (x_mid[i+1] - x_mid[i])
    print(f"Интервал [{x_mid[i]:.2f}, {x_mid[i+1]:.2f}]: y'' = {d2y:.2f}")

# 2. Численное интегрирование
def f(x):
    return 4**x

a, b = 1.0, 5.0
n = 1000

# Метод средних прямоугольников
h = (b - a) / n
integral = sum(h * f(a + h/2 + i*h) for i in range(n))
print(f"\nМетод средних прямоугольников: ∫4^x dx от {a} до {b} ≈ {integral:.2f}")

# Метод трапеций
integral = 0.5*(f(a) + f(b)) + sum(f(a + i*h) for i in range(1, n))
integral *= h
print(f"Метод трапеций: ∫4^x dx от {a} до {b} ≈ {integral:.2f}")

# Метод Симпсона
integral = f(a) + f(b)
for i in range(1, n):
    coeff = 4 if i % 2 else 2
    integral += coeff * f(a + i * h)
integral *= h/3
print(f"Метод Симпсона: ∫4^x dx от {a} до {b} ≈ {integral:.2f}")

# Точное значение интеграла
exact = (4**b - 4**a) / np.log(4)
print(f"Точное значение: ∫4^x dx от {a} до {b} = {exact:.2f}")