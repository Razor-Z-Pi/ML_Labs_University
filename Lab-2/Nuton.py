import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def f(x):
    if abs(5 * x**2 + x) > 1e10:
        return 0
    else:
        return math.cos(5 * x**2 + x) + x**3 + 5 * x
    
def f_proiz(x):
    return -math.sin(5 * x**2 + x) * (10 * x + 1) + 3 * x**2 + 5

# Метод Ньютона
def newton(f, f_proiz, x0, toleration=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fpx = f_proiz(x)
        if abs(fpx) < toleration:
            break
        x_new = x - fx / fpx
        if abs(x_new - x) < toleration:
            return x_new
        x = x_new
    return x

# Метод хорд
def chord(f, a, b, toleration=1e-6, max_iter=100):
    for i in range(max_iter):
        fa = f(a)
        fb = f(b)
        x_new = (a * fb - b * fa) / (fb - fa)
        if abs(f(x_new)) < toleration:
            return x_new
        if f(a) * f(x_new) < 0:
            b = x_new
        else:
            a = x_new
    return x_new

# Метод итерации
def iter(f, x0, toleration=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = f(x) + x 
        if abs(x_new - x) < toleration:
            return x_new
        x = x_new
    return x


x_values = [i / 100 for i in range(-200, 200)]
y_values = [f(x) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="f(x) = cos(5x^2 + x) + x^3 + 5x")
plt.axhline(0, color="black", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("График функции f(x)")
plt.legend()
plt.grid()
plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", name="f(x)"))
fig.update_layout(
    title="График функции f(x)",
    xaxis_title="x",
    yaxis_title="f(x)",
    template="plotly_white",
)
fig.show()

lister = [-1.5, -0.5, 0.5, 1.5]

print("Метод Ньютона:")
for x0 in lister:
    root = newton(f, f_proiz, x0)
    print(f"Начальное приближение: {x0}, Корень: {root}, f(x) = {f(root)}")


print("\nМетод хорд:")
intervals = [(-2, -1), (-1, 0), (0, 1), (1, 2)]
for a, b in intervals:
    root = chord(f, a, b)
    print(f"Интервал: [{a}, {b}], Корень: {root}, f(x) = {f(root)}")

print("\nМетод итерации:")
for x0 in lister:
    root = iter(f, x0)
    print(f"Начальное приближение: {x0}, Корень: {root}, f(x) = {f(root)}")