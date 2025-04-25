import os
import matplotlib.pyplot as plt
from keras import Sequential, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import mnist
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras.regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# --- Блок параметров для борьбы с переобучением ---
L1_REGULARIZATION = 0.00005
L2_REGULARIZATION = 0.00005
USE_REGULARIZATION = True
DROPOUT_RATE = 0.4

# --- Параметры обучения ---
EPOCH_COUNT = 400  # Увеличим количество эпох
BATCH_SIZE = 128  # Увеличим размер пакета
LEARNING_RATE = 0.001

# --- Параметры аугментации данных ---
USE_DATA_AUGMENTATION = True
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.15
HEIGHT_SHIFT_RANGE = 0.15
ZOOM_RANGE = 0.15
SHEAR_RANGE = 0.1  # Добавим сдвиг
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'

# --- Параметры Early Stopping и Learning Rate Reduction ---
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_MONITOR = 'val_loss'
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 5
REDUCE_LR_MIN_LR = 1e-6

# --- Блок параметров для путей и версий ---
RESULTS_DIR = 'results'
MODEL_VERSION = 5.1  # Обновляем версию модели
MODEL_SUBDIR = f'model_cnn_v{MODEL_VERSION}'
OUTPUT_DIR = os.path.join(RESULTS_DIR, MODEL_SUBDIR)

# Создание директорий
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Определение путей внутри папки версии
MODEL_PATH = os.path.join("mnist_model", 'mnist_cnn_model_v'+str(MODEL_VERSION)+'.keras')
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
    train_images, train_labels, test_size=0.1, random_state=42
)

# Подготовка входных данных для сверточных слоев
train_images_cnn = np.expand_dims(train_images, -1)
val_images_cnn = np.expand_dims(val_images, -1)
test_images_cnn = np.expand_dims(test_images, -1)

# Создание и компиляция сверточной модели
regularize = None
if USE_REGULARIZATION:
    regularize = keras.regularizers.L1L2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)
    print(f"Используется регуляризация L1={L1_REGULARIZATION}, L2={L2_REGULARIZATION}")
else:
    print("Регуляризация не используется.")

model_cnn = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularize),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularize),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(DROPOUT_RATE),

    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularize),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularize),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(DROPOUT_RATE),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularize),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(DROPOUT_RATE),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularize),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),
    Dense(10, activation='softmax')
])

optimizer = Adam(learning_rate=LEARNING_RATE)
model_cnn.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# EarlyStopping
early_stopping = EarlyStopping(
    monitor=EARLY_STOPPING_MONITOR,
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)

# Reduce Learning Rate on Plateau
reduce_lr = ReduceLROnPlateau(
    monitor=REDUCE_LR_MONITOR,
    factor=REDUCE_LR_FACTOR,
    patience=REDUCE_LR_PATIENCE,
    min_lr=REDUCE_LR_MIN_LR,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# Аугментация данных
if USE_DATA_AUGMENTATION:
    datagen = ImageDataGenerator(
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        shear_range=SHEAR_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode=FILL_MODE
    )
    datagen.fit(train_images_cnn)

    print("Используется аугментация данных.")
    history = model_cnn.fit(datagen.flow(train_images_cnn, train_labels, batch_size=BATCH_SIZE),
                            epochs=EPOCH_COUNT,
                            validation_data=(val_images_cnn, val_labels),
                            callbacks=callbacks,
                            verbose=1)
else:
    print("Аугментация данных не используется.")
    history = model_cnn.fit(train_images_cnn,
                            train_labels,
                            epochs=EPOCH_COUNT,
                            batch_size=BATCH_SIZE,
                            validation_data=(val_images_cnn, val_labels),
                            callbacks=callbacks,
                            verbose=1)

# Сохранение CNN модели
model_cnn.save(MODEL_PATH)
print(f"CNN модель сохранена в: {MODEL_PATH}")

# Оценка CNN модели
loss, accuracy = model_cnn.evaluate(test_images_cnn, test_labels, verbose=0)
print(f"Тестовые потери (CNN): {loss:.4f}")
print(f"Тестовая точность (CNN): {accuracy:.4f}")

# Извлечение данных для графиков
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Сохранение точности в файл
with open(ACCURACY_PATH, 'w') as f:
    f.write('Точность обучения:\n')
    for acc in accuracy:
        f.write(str(acc) + '\n')
    f.write('\nТочность валидации:\n')
    for acc in val_accuracy:
        f.write(str(acc) + '\n')
    f.write('\nПотери обучения:\n')
    for l in loss:
        f.write(str(l) + '\n')
    f.write('\nПотери валидации:\n')
    for l in val_loss:
        f.write(str(l) + '\n')
print(f"Данные о точности и потерях сохранены в: {ACCURACY_PATH}")

# Построение графика точности
epochs = range(1, len(accuracy) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'bo', label='Точность обучения')
plt.plot(epochs, val_accuracy, 'b', label='Точность валидации')
plt.title('Точность обучения и валидации (CNN)')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)

# Построение графика потерь
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'ro', label='Потери обучения')
plt.plot(epochs, val_loss, 'r', label='Потери валидации')
plt.title('Потери обучения и валидации (CNN)')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

plt.savefig(PLOT_PATH)
plt.show()
print(f"График точности и потерь сохранен в: {PLOT_PATH}")

# Confusion Matrix
predictions = model_cnn.predict(test_images_cnn)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)
cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (CNN)")
plt.xlabel("Предсказанные метки")
plt.ylabel("Истинные метки")
plt.savefig(CM_PATH)
plt.show()
print(f"Confusion Matrix сохранен в: {CM_PATH}")

# Вывод финальной точности
print(f"Финальная точность обучения (CNN): {accuracy[-1]:.4f}")
print(f"Финальная точность валидации (CNN): {val_accuracy[-1]:.4f}")