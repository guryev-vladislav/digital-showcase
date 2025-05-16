import pandas as pd
import matplotlib.pyplot as plt

# Чтение данных из CSV
try:
    spline_data = pd.read_csv("spline_data.csv")
    original_data = pd.read_csv("original_data.csv")
except FileNotFoundError:
    print("Ошибка: Файл spline_data.csv или original_data.csv не найден.")
    exit()

# Визуализация
plt.figure(figsize=(12, 6))

# График функции и сплайна
plt.subplot(1, 2, 1)
plt.plot(original_data['x'], original_data['f_x'], 'bo', label='Исходные точки')
plt.plot(spline_data['x'], spline_data['s_x'], 'r-', label='Сплайн')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация сплайном')
plt.legend()
plt.grid(True)

# График погрешности
plt.subplot(1, 2, 2)
plt.plot(original_data['x'], original_data['pogr'], 'g-', label='Погрешность')
plt.xlabel('x')
plt.ylabel('Погрешность')
plt.title('Погрешность аппроксимации')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Таблица (вывод в консоль)
print("\nТаблица исходных данных:")
print(original_data[['x', 'f_x']])

print("\nТаблица значений сплайна:")
print(spline_data[['x', 's_x']])

print("\nТаблица погрешностей:")
print(original_data[['x', 'pogr']])

# Нахождение максимальной погрешности
max_pogr_value = original_data['pogr'].max()
max_pogr_x = original_data['x'][original_data['pogr'].idxmax()]
print(f"\nМаксимальная погрешность: {max_pogr_value} при x = {max_pogr_x}")