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
import datetime  # Для уникальной временной метки запуска

# --- Блок параметров для борьбы с переобучением ---
L1_REGULARIZATION = 0.0001  # Значение L1 регуляризации
L2_REGULARIZATION = 0.0001  # Значение L2 регуляризации
USE_REGULARIZATION = True  # Использовать регуляризацию или нет
DROPOUT_RATE = 0.2

# Параметры EarlyStopping
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True  # Добавлено для явности

EPOCH_COUNT = 150  # Максимальное количество эпох

# --- Блок параметров для путей и версий ---
RESULTS_FOLDER = 'results'  # Общая папка для всех результатов
MODELS_FOLDER = 'models'  # Общая папка для сохранения всех обученных моделей (ранее 'mnist_models')


# --- Функции для сохранения результатов ---
def plot_confusion_matrix(y_true, y_pred, classes, title, filename, save_dir):
    """
    Строит и сохраняет матрицу ошибок.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()  # Закрываем фигуру, чтобы не отображалась сразу
    print(f"Матрица ошибок сохранена в: {os.path.join(save_dir, filename)}")


def save_run_parameters(params, save_dir, filename='parameters.txt'):
    """
    Сохраняет все важные параметры запуска в текстовый файл.
    """
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        f.write("--- Run Parameters ---\n")
        f.write(f"Run Timestamp: {params.get('RUN_TIMESTAMP', 'N/A')}\n\n")

        f.write("--- Dataset Information ---\n")
        f.write(f"Train Samples: {params.get('TRAIN_SAMPLES', 'N/A')}\n")
        f.write(f"Validation Samples: {params.get('VAL_SAMPLES', 'N/A')}\n")
        f.write(f"Test Samples: {params.get('TEST_SAMPLES', 'N/A')}\n")
        f.write("\n")

        f.write("--- Model Architecture and Parameters ---\n")
        f.write(f"Model Type: Simple Dense Network for MNIST\n")  # Уточняем тип модели
        f.write(f"Input Shape: {params.get('INPUT_SHAPE', 'N/A')}\n")
        f.write(f"Number of Classes: {params.get('NUM_CLASSES', 'N/A')}\n")
        f.write(f"Total Model Parameters: {params.get('MODEL_TOTAL_PARAMS', 'N/A')}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Loss Function: Categorical Crossentropy\n")
        f.write(f"Metrics: Accuracy\n")

        f.write(f"L1 Regularization: {params.get('L1_REGULARIZATION', 'N/A')}\n")
        f.write(f"L2 Regularization: {params.get('L2_REGULARIZATION', 'N/A')}\n")
        f.write(f"Use Regularization: {params.get('USE_REGULARIZATION', 'N/A')}\n")
        f.write(f"Dropout Rate: {params.get('DROPOUT_RATE', 'N/A')}\n")
        f.write("\n")

        f.write("--- Training Parameters ---\n")
        f.write(f"Max Epochs: {params.get('EPOCH_COUNT', 'N/A')}\n")
        f.write(f"Batch Size: {params.get('BATCH_SIZE', 'N/A')}\n")
        f.write(f"Early Stopping Monitor: {params.get('EARLY_STOPPING_MONITOR', 'N/A')}\n")
        f.write(f"Early Stopping Patience: {params.get('EARLY_STOPPING_PATIENCE', 'N/A')}\n")
        f.write(f"Early Stopping Restore Best Weights: {params.get('EARLY_STOPPING_RESTORE_BEST_WEIGHTS', 'N/A')}\n")
        f.write("\n")

        f.write("--- Final Metrics (Training History) ---\n")
        f.write(f"Final Training Loss: {params.get('FINAL_TRAINING_LOSS', 'N/A'):.4f}\n")
        f.write(f"Final Validation Loss: {params.get('FINAL_VALIDATION_LOSS', 'N/A'):.4f}\n")
        f.write(f"Final Training Accuracy: {params.get('FINAL_TRAINING_ACCURACY', 'N/A'):.4f}\n")
        f.write(f"Final Validation Accuracy: {params.get('FINAL_VALIDATION_ACCURACY', 'N/A'):.4f}\n")
        f.write("\n")

        f.write("--- Test Metrics ---\n")
        f.write(f"Test Loss: {params.get('TEST_LOSS', 'N/A'):.4f}\n")
        f.write(f"Test Accuracy: {params.get('TEST_ACCURACY', 'N/A'):.4f}\n")
        f.write("\n")

        f.write(f"Model Save Path: {params.get('MODEL_SAVE_PATH', 'N/A')}\n")
        f.write(f"Results Directory: {save_dir}\n")

    print(f"Параметры запуска сохранены в: {filepath}")


# --- Основная логика запуска ---
if __name__ == '__main__':
    # Генерация уникальной директории для текущего запуска
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Добавляем префикс 'cnn_' к имени папки, чтобы отличить от других типов моделей
    current_run_dir = os.path.join(RESULTS_FOLDER, f'cnn_{run_timestamp}')
    os.makedirs(current_run_dir, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)  # Убедимся, что общая папка для моделей существует

    # Определение путей для текущего запуска
    # Модель сохраняется в общую папку MODELS_FOLDER с префиксом cnn_
    MODEL_PATH_CURRENT_RUN = os.path.join(MODELS_FOLDER, f'cnn_mnist_model_{run_timestamp}.keras')
    ACCURACY_PLOT_PATH = os.path.join(current_run_dir, 'accuracy_plot.png')
    CONFUSION_MATRIX_PATH = os.path.join(current_run_dir, 'confusion_matrix.png')
    PARAMETERS_FILE_PATH = os.path.join(current_run_dir, 'parameters.txt')  # Путь к файлу параметров

    print(f"Результаты текущего запуска будут сохранены в: {current_run_dir}")

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
    # Определение размера тестового и валидационного набора для сохранения в параметры
    TEST_SIZE_SPLIT = 0.1  # Размер тестового набора MNIST изначальный (10000)
    VAL_SIZE_SPLIT_FROM_TRAIN = 0.1  # test_size в train_test_split от исходного train_images (60000)

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=VAL_SIZE_SPLIT_FROM_TRAIN, random_state=42)  # Это от 60000 образцов

    # Определяем количество образцов после разбиения
    TRAIN_SAMPLES_COUNT = len(train_images)
    VAL_SAMPLES_COUNT = len(val_images)
    TEST_SAMPLES_COUNT = len(test_images)  # test_images уже является отдельным набором

    # Input Shape для параметров: (высота, ширина, каналы). Для MNIST это (28, 28, 1)
    # Хотя Flatten преобразует его в 1D, исходная форма данных важна для понимания.
    INPUT_SHAPE_MODEL = (28, 28, 1)

    # Создание и компиляция модели
    regularize = None
    if USE_REGULARIZATION:
        regularize = keras.regularizers.L1L2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)
        print(f"Используется регуляризация L1={L1_REGULARIZATION}, L2={L2_REGULARIZATION}")
    else:
        print("Регуляризация не используется.")

    model = Sequential([
        Input(shape=(28, 28)),  # Входной слой для 2D-изображений
        Flatten(),  # Преобразует 2D-изображения в 1D-вектор
        Dense(128, activation='relu', kernel_regularizer=regularize),  # Добавлена регуляризация
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu', kernel_regularizer=regularize),  # Добавлена регуляризация
        Dropout(DROPOUT_RATE),
        Dense(10, activation='softmax')  # Выходной слой для 10 классов MNIST
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()  # Выводим сводку модели

    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor=EARLY_STOPPING_MONITOR,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS
    )

    # Обучение модели
    print("Начинаем обучение модели...")
    # Добавляем batch_size в .fit() для явности, т.к. это параметр обучения
    BATCH_SIZE_TRAINING = 32
    history = model.fit(train_images,
                        train_labels,
                        epochs=EPOCH_COUNT,
                        batch_size=BATCH_SIZE_TRAINING,
                        validation_data=(val_images, val_labels),
                        callbacks=[early_stopping],
                        verbose=1)
    print("Обучение завершено.")

    # Сохранение модели
    model.save(MODEL_PATH_CURRENT_RUN)
    print(f"Модель сохранена в: {MODEL_PATH_CURRENT_RUN}")

    # Оценка модели
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Тестовые потери: {test_loss:.4f}")
    print(f"Тестовая точность: {test_accuracy:.4f}")

    # Извлечение данных для графиков
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Построение графика точности и потерь
    epochs_ran = range(1, len(accuracy) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_ran, accuracy, 'bo', label='Точность обучения')
    plt.plot(epochs_ran, val_accuracy, 'b', label='Точность валидации')
    plt.title('Точность обучения и валидации')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, loss, 'ro', label='Потери обучения')
    plt.plot(epochs_ran, val_loss, 'r', label='Потери валидации')
    plt.title('Потери обучения и валидации')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH)
    plt.show()
    plt.close()  # Закрываем фигуру
    print(f"График точности и потерь сохранен в: {ACCURACY_PLOT_PATH}")

    # Confusion Matrix
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # Классы для матрицы ошибок (0-9)
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(true_labels, predicted_labels, class_names,
                          "Confusion Matrix (Test Data)", "confusion_matrix.png", current_run_dir)

    # Вывод финальной точности
    final_train_accuracy = accuracy[-1]
    final_val_accuracy = val_accuracy[-1]
    final_train_loss = loss[-1]
    final_val_loss = val_loss[-1]

    print(f"Финальная точность обучения: {final_train_accuracy:.4f}")
    print(f"Финальная точность валидации: {final_val_accuracy:.4f}")

    # Сохранение параметров запуска в файл
    run_parameters = {
        "RUN_TIMESTAMP": run_timestamp,
        "TRAIN_SAMPLES": TRAIN_SAMPLES_COUNT,
        "VAL_SAMPLES": VAL_SAMPLES_COUNT,
        "TEST_SAMPLES": TEST_SAMPLES_COUNT,
        "INPUT_SHAPE": str(INPUT_SHAPE_MODEL),
        "NUM_CLASSES": 10,  # Для MNIST всегда 10 классов
        "MODEL_TOTAL_PARAMS": model.count_params(),
        "L1_REGULARIZATION": L1_REGULARIZATION,
        "L2_REGULARIZATION": L2_REGULARIZATION,
        "USE_REGULARIZATION": USE_REGULARIZATION,
        "DROPOUT_RATE": DROPOUT_RATE,
        "EPOCH_COUNT": EPOCH_COUNT,
        "BATCH_SIZE": BATCH_SIZE_TRAINING,  # Используем явно заданный batch_size
        "EARLY_STOPPING_MONITOR": EARLY_STOPPING_MONITOR,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "EARLY_STOPPING_RESTORE_BEST_WEIGHTS": EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
        "FINAL_TRAINING_LOSS": final_train_loss,
        "FINAL_VALIDATION_LOSS": final_val_loss,
        "FINAL_TRAINING_ACCURACY": final_train_accuracy,
        "FINAL_VALIDATION_ACCURACY": final_val_accuracy,
        "TEST_LOSS": test_loss,
        "TEST_ACCURACY": test_accuracy,
        "MODEL_SAVE_PATH": MODEL_PATH_CURRENT_RUN,
        "RESULTS_DIRECTORY": current_run_dir,
    }

    save_run_parameters(run_parameters, current_run_dir, filename='parameters.txt')

    print("Все задачи выполнены. Результаты сохранены в папке 'results'.")