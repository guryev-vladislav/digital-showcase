import os
import matplotlib.pyplot as plt
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.datasets import mnist
from keras.src.layers import Flatten, Dense, Dropout
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras.src.regularizers

# --- Блок параметров для борьбы с переобучением ---
L1_REGULARIZATION = 0.0001  # Значение L1 регуляризации
L2_REGULARIZATION = 0.0001  # Значение L2 регуляризации
USE_REGULARIZATION = True  # Использовать регуляризацию или нет
DROPOUT_RATE = 0.2
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
EPOCH_COUNT = 150

# --- Блок параметров для путей и версий ---
RESULTS_DIR = 'results'
MODEL_VERSION = 4.4  # Увеличиваем версию, чтобы не перезаписать предыдущие результаты
MODEL_SUBDIR = f'model_v{MODEL_VERSION}'
OUTPUT_DIR = os.path.join(RESULTS_DIR, MODEL_SUBDIR)

# Создание директорий
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Определение путей внутри папки версии
MODEL_PATH = os.path.join("mnist_model", 'mnist_model_v'+str(MODEL_VERSION)+'.keras')
ACCURACY_PATH = os.path.join(OUTPUT_DIR, 'accuracy.txt')
PLOT_PATH = os.path.join(OUTPUT_DIR, 'accuracy_plot.png')
CM_PATH = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')

# Загрузка данных с обработкой ошибок
try:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print("Данные MNIST успешно загружены.")
except FileNotFoundError:
    print("Ошибка: Не удалось загрузить данные MNIST.")
    exit()
except Exception as e:
    print(f"Произошла ошибка при загрузке данных: {e}")
    exit()

# Нормализация и One-hot encoding
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Разделение на обучающую и валидационную выборки
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42)

# Создание и компиляция модели
regularize = None
if USE_REGULARIZATION:
    regularize = keras.regularizers.L1L2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)
    print(f"Используется регуляризация L1={L1_REGULARIZATION}, L2={L2_REGULARIZATION}")
else:
    print("Регуляризация не используется.")
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularize),  # Добавлена регуляризация
    Dropout(DROPOUT_RATE),
    Dense(64, activation='relu', kernel_regularizer=regularize),  # Добавлена регуляризация
    Dropout(DROPOUT_RATE),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping
early_stopping = EarlyStopping(
    monitor=EARLY_STOPPING_MONITOR,
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)

# Обучение модели
print("Начинаем обучение модели...")
history = model.fit(train_images,
                    train_labels,
                    epochs=EPOCH_COUNT,
                    validation_data=(val_images, val_labels),
                    callbacks=[early_stopping],
                    verbose=1)

# Сохранение модели
model.save(MODEL_PATH)
print(f"Модель сохранена в: {MODEL_PATH}")

# Оценка модели
loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Тестовые потери: {loss:.4f}")
print(f"Тестовая точность: {accuracy:.4f}")

# Извлечение данных для графиков
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Сохранение точности в файл
with open(ACCURACY_PATH, 'w') as f:
    f.write('Точность обучения:\n')
    for acc in accuracy:
        f.write(str(acc) + '\n')
    f.write('\nТочность валидации:\n')
    for acc in val_accuracy:
        f.write(str(acc) + '\n')
print(f"Данные о точности сохранены в: {ACCURACY_PATH}")

# Построение графика точности
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'bo', label='Точность обучения')
plt.plot(epochs, val_accuracy, 'b', label='Точность валидации')
plt.title('Точность обучения и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.savefig(PLOT_PATH)
plt.show()
print(f"График точности сохранен в: {PLOT_PATH}")

# Confusion Matrix
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Предсказанные метки")
plt.ylabel("Истинные метки")
plt.savefig(CM_PATH)
plt.show()
print(f"Confusion Matrix сохранен в: {CM_PATH}")

# Вывод финальной точности
print(f"Финальная точность обучения: {accuracy[-1]:.4f}")
print(f"Финальная точность валидации: {val_accuracy[-1]:.4f}")