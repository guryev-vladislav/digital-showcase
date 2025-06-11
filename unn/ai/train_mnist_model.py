import os
import matplotlib.pyplot as plt
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau  # ReduceLROnPlateau не используется, но оставлен
from keras.src.datasets import mnist
from keras.src.layers import Flatten, Dense, Dropout
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras.src.regularizers
import datetime  # Для уникальной временной метки запуска
import h5py  # Для работы с HDF5 файлами

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

# --- Блок параметров для выбора датасета ---
USE_GENERATED_DATASET = True  # Установите True, чтобы использовать HDF5 датасет; False для стандартного MNIST

# --- Параметры для загрузки из HDF5 ---
DATASET_FOLDER = 'datasets'
# Убедитесь, что это имя файла соответствует тому, что вы сгенерировали
DATASET_FILENAME = 'synthetic_mnist_large_scale_flexible_sizes_v5000.h5'
H5_FILE_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

# --- Блок параметров для путей и версий ---
RESULTS_FOLDER = 'results'  # Общая папка для всех результатов
MODELS_FOLDER = 'models'  # Общая папка для сохранения всех обученных моделей


# --- Функции для загрузки данных из HDF5 ---
def load_data_from_h5(h5_file_path):
    """
    Загружает изображения и метки из HDF5 файла.
    Для этой модели нам нужны только x_*, y_*.
    """
    print(f"Загрузка данных из HDF5 файла: {h5_file_path}...")
    with h5py.File(h5_file_path, 'r') as f:
        x_train = np.array(f['x_train'])
        y_train = np.array(f['y_train'])
        x_val = np.array(f['x_val'])
        y_val = np.array(f['y_val'])
        x_test = np.array(f['x_test'])
        y_test = np.array(f['y_test'])
    print("Данные успешно загружены из HDF5.")
    # Возвращаем все как (обучающие_данные, валидационные_данные, тестовые_данные) для единообразия
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


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
    plt.show()  # Опционально, чтобы отобразить, если запускается интерактивно
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
        f.write(f"Dataset Source: {params.get('DATASET_SOURCE', 'N/A')}\n")
        if params.get('USE_GENERATED_DATASET'):  # Используем флаг из параметров для логирования
            f.write(f"Dataset Filename: {params.get('DATASET_FILENAME', 'N/A')}\n")
            f.write(f"H5 File Path: {params.get('H5_FILE_PATH', 'N/A')}\n")
        f.write(f"Train Samples: {params.get('TRAIN_SAMPLES', 'N/A')}\n")
        f.write(f"Validation Samples: {params.get('VAL_SAMPLES', 'N/A')}\n")
        f.write(f"Test Samples: {params.get('TEST_SAMPLES', 'N/A')}\n")
        f.write("\n")

        f.write("--- Model Architecture and Parameters ---\n")
        f.write(f"Model Type: Simple Dense Network for MNIST\n")
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
    current_run_dir = os.path.join(RESULTS_FOLDER, f'cnn_{run_timestamp}')
    os.makedirs(current_run_dir, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)  # Убедимся, что папка для моделей существует

    # Определение путей для текущего запуска
    MODEL_PATH_CURRENT_RUN = os.path.join(MODELS_FOLDER, f'cnn_mnist_model_{run_timestamp}.keras')
    ACCURACY_PLOT_PATH = os.path.join(current_run_dir, 'accuracy_plot.png')
    CONFUSION_MATRIX_PATH = os.path.join(current_run_dir, 'confusion_matrix.png')
    PARAMETERS_FILE_PATH = os.path.join(current_run_dir, 'parameters.txt')  # Путь к файлу параметров

    print(f"Результаты текущего запуска будут сохранены в: {current_run_dir}")

    # --- Загрузка данных ---
    if USE_GENERATED_DATASET:
        dataset_source_name = "Generated HDF5"
        (x_train_raw, y_train_raw), \
            (x_val_raw, y_val_raw), \
            (x_test_raw, y_test_raw) = load_data_from_h5(H5_FILE_PATH)

        # Данные из HDF5 уже предположительно нормализованы к [0, 1] и имеют размерность (H, W, 1)
        train_images = x_train_raw
        val_images = x_val_raw
        test_images = x_test_raw

        # HDF5-файл уже содержит валидационный набор, поэтому train_test_split не нужен для создания val
        train_labels_raw = y_train_raw
        val_labels_raw = y_val_raw
        test_labels_raw = y_test_raw

        # Если метки из HDF5 не в one-hot формате, преобразовать их
        if len(train_labels_raw.shape) == 1:  # Проверяем, если не one-hot (одномерный массив)
            train_labels = to_categorical(train_labels_raw)
            val_labels = to_categorical(val_labels_raw)
            test_labels = to_categorical(test_labels_raw)
        else:  # Если уже one-hot (например, (N, 10))
            train_labels = train_labels_raw
            val_labels = val_labels_raw
            test_labels = test_labels_raw

    else:  # Используем стандартный MNIST
        dataset_source_name = "Standard MNIST"
        try:
            (train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()
            print("Данные MNIST успешно загружены.")
        except FileNotFoundError:
            print("Ошибка: Не удалось загрузить данные MNIST.")
            exit()
        except Exception as e:
            print(f"Произошла ошибка при загрузке данных: {e}")
            exit()

        # Нормализация
        train_images_mnist = train_images_mnist / 255.0
        test_images_mnist = test_images_mnist / 255.0

        # One-hot encoding
        train_labels_mnist = to_categorical(train_labels_mnist)
        test_labels_mnist = to_categorical(test_labels_mnist)

        # Разделение на обучающую и валидационную выборки (для стандартного MNIST)
        VAL_SIZE_SPLIT_FROM_TRAIN = 0.1  # test_size в train_test_split от исходного train_images
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images_mnist, train_labels_mnist, test_size=VAL_SIZE_SPLIT_FROM_TRAIN, random_state=42)

        test_images = test_images_mnist
        test_labels = test_labels_mnist

    # Определяем количество образцов после загрузки и разбиения
    TRAIN_SAMPLES_COUNT = len(train_images)
    VAL_SAMPLES_COUNT = len(val_images)
    TEST_SAMPLES_COUNT = len(test_images)

    # --- Автоматическое определение INPUT_SHAPE для модели ---
    # Если данные имеют 3 измерения (H, W), как 28x28 из MNIST, добавляем измерение канала
    if len(train_images.shape) == 3:
        train_images = np.expand_dims(train_images, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)

    # Форма входных данных для слоя Input (без batch_size и без канала, т.к. Flatten его "сплющит")
    # Например, для (N, 28, 28, 1) это будет (28, 28)
    # Для (N, 112, 112, 1) это будет (112, 112)
    input_shape_for_model_input = train_images.shape[1:-1]

    print(f"Итоговые размеры данных для обучения:")
    print(f"train_images: {train_images.shape}, train_labels: {train_labels.shape}")
    print(f"val_images:   {val_images.shape}, val_labels:   {val_labels.shape}")
    print(f"test_images:  {test_images.shape}, test_labels:  {test_labels.shape}")
    print(f"Автоматически определенная форма входных данных для слоя Input: {input_shape_for_model_input}")

    # Создание и компиляция модели
    regularize = None
    if USE_REGULARIZATION:
        regularize = keras.regularizers.L1L2(l1=L1_REGULARIZATION, l2=L2_REGULARIZATION)
        print(f"Используется регуляризация L1={L1_REGULARIZATION}, L2={L2_REGULARIZATION}")
    else:
        print("Регуляризация не используется.")

    # --- Input(shape=...) теперь определяется автоматически ---
    model = Sequential([
        Input(shape=input_shape_for_model_input),  # Теперь Input(shape=...) автоматически подстраивается
        Flatten(),
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
    BATCH_SIZE_TRAINING = 32  # Указываем явно batch_size
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
        "DATASET_SOURCE": dataset_source_name,
        "USE_GENERATED_DATASET": USE_GENERATED_DATASET,  # Флаг для логирования
        "DATASET_FILENAME": DATASET_FILENAME if USE_GENERATED_DATASET else "N/A",
        "H5_FILE_PATH": H5_FILE_PATH if USE_GENERATED_DATASET else "N/A",
        "TRAIN_SAMPLES": TRAIN_SAMPLES_COUNT,
        "VAL_SAMPLES": VAL_SAMPLES_COUNT,
        "TEST_SAMPLES": TEST_SAMPLES_COUNT,
        "INPUT_SHAPE": str(input_shape_for_model_input),  # Сохраняем автоматически определенную форму
        "NUM_CLASSES": 10,  # Для MNIST всегда 10 классов
        "MODEL_TOTAL_PARAMS": model.count_params(),
        "L1_REGULARIZATION": L1_REGULARIZATION,
        "L2_REGULARIZATION": L2_REGULARIZATION,
        "USE_REGULARIZATION": USE_REGULARIZATION,
        "DROPOUT_RATE": DROPOUT_RATE,
        "EPOCH_COUNT": EPOCH_COUNT,
        "BATCH_SIZE": BATCH_SIZE_TRAINING,
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