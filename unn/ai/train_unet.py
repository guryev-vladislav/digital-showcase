import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

# --- КОНФИГУРАЦИЯ ФАЙЛОВ И ДАТАСЕТА ---
DATASET_FOLDER = 'datasets'
DATASET_FILENAME = 'synthetic_mnist_large_scale_flexible_sizes_v5000.h5'
H5_FILE_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'

# --- ПАРАМЕТРЫ МОДЕЛИ И ОБУЧЕНИЯ ---
INPUT_SHAPE = (112, 112, 1)  # Размер входного изображения (высота, ширина, каналы)
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Параметры EarlyStopping
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True

# Параметры ReduceLROnPlateau
REDUCE_LR_MONITOR = 'val_loss'
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_LR = 1e-6


# --- Загрузка данных из HDF5 ---
def load_data(h5_file_path):
    print(f"Загрузка данных из HDF5 файла: {h5_file_path}...")
    with h5py.File(h5_file_path, 'r') as f:
        # Загрузка обучающих данных
        print("Загрузка обучающих данных:")
        x_train = np.array(f['x_train'])
        mask_train = np.array(f['mask_train'])
        y_train = np.array(f['y_train'])
        print("Загрузка валидационных данных:")
        x_val = np.array(f['x_val'])
        mask_val = np.array(f['mask_val'])
        y_val = np.array(f['y_val'])
        print("Загрузка тестовых данных:")
        x_test = np.array(f['x_test'])
        mask_test = np.array(f['mask_test'])
        y_test = np.array(f['y_test'])
    print("Данные успешно загружены.")
    print(f"Размеры загруженных данных:")
    print(f"x_train: {x_train.shape}, mask_train: {mask_train.shape}, y_train: {y_train.shape}")
    print(f"x_val:   {x_val.shape}, mask_val:   {mask_val.shape}, y_val:   {y_val.shape}")
    print(f"x_test:  {x_test.shape}, mask_test:  {mask_test.shape}, y_test:  {y_test.shape}")
    return (x_train, mask_train, y_train), \
        (x_val, mask_val, y_val), \
        (x_test, mask_test, y_test)


# --- U-Net Модель ---
def build_unet(input_shape, num_classes):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # (56, 56, 32)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # (28, 28, 64)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)  # (14, 14, 128)

    # Bottleneck
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)  # (14, 14, 256)

    # Classification Branch (uses features from bottleneck)
    global_pool = layers.GlobalAveragePooling2D()(conv4)
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification_output')(global_pool)

    # Decoder
    up6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    merge6 = layers.concatenate([conv3, up6], axis=3)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)  # (28, 28, 128)

    up7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv2, up7], axis=3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)  # (56, 56, 64)

    up8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv1, up8], axis=3)
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv8)  # (112, 112, 32)

    # Segmentation Output
    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation_output')(conv8)

    model = models.Model(inputs=inputs, outputs=[segmentation_output, classification_output])

    return model


# --- Функция для создания tf.data.Dataset ---
def create_dataset(images, masks, labels, batch_size, shuffle=True):
    # Преобразование в tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, masks, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))

    # Маппинг для соответствия мультивыходной модели
    # inputs = image, outputs = {'segmentation_output': mask, 'classification_output': label}
    dataset = dataset.map(lambda img, msk, lbl: (img, {'segmentation_output': msk, 'classification_output': lbl}),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# --- Визуализация истории обучения ---
def plot_training_history(history, save_dir):
    plt.figure(figsize=(18, 6))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Total Training Loss')
    plt.plot(history.history['val_loss'], label='Total Validation Loss')
    plt.plot(history.history['segmentation_output_loss'], label='Segmentation Training Loss')
    plt.plot(history.history['val_segmentation_output_loss'], label='Segmentation Validation Loss')
    plt.plot(history.history['classification_output_loss'], label='Classification Training Loss')
    plt.plot(history.history['val_classification_output_loss'], label='Classification Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Установка лимитов Y для графика потерь
    min_loss = min(min(history.history['loss']), min(history.history['val_loss']), 0)
    max_loss = max(max(history.history['loss']), max(history.history['val_loss']))
    plt.ylim(max(0, min_loss * 0.9), max_loss * 1.1)  # Автоматический лимит с небольшим запасом

    # Plot Segmentation Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['segmentation_output_accuracy'], label='Segmentation Training Accuracy')
    plt.plot(history.history['val_segmentation_output_accuracy'], label='Segmentation Validation Accuracy')
    plt.title('Segmentation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    # Установка лимитов Y для точности сегментации, так как она очень высока
    plt.ylim(0.985, 1.0)  # Задаем более точный диапазон, так как точность близка к 1

    # Plot Classification Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(history.history['classification_output_accuracy'], label='Classification Training Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Classification Validation Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    # Установка лимитов Y для точности классификации
    plt.ylim(0, 1.0)  # Диапазон от 0 до 1, так как точность может быть в любом месте

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()  # Закрываем фигуру
    plt.show()  # Опционально, чтобы отобразить, если запускается интерактивно


# --- Визуализация предсказаний ---
def visualize_predictions(model, images, true_masks, true_labels, num_samples=5, save_dir=None, prefix=""):
    plt.figure(figsize=(15, 10))

    # Выбираем случайные индексы для визуализации
    indices = np.random.choice(len(images), num_samples, replace=False)

    for i, idx in enumerate(indices):
        # Получаем один пример и добавляем измерение для батча
        single_image = np.expand_dims(images[idx], axis=0)
        true_mask = true_masks[idx]
        true_label = np.argmax(true_labels[idx])

        # Делаем предсказание
        predicted_mask_batch, predicted_label_batch = model.predict(single_image, verbose=0)

        # Убираем измерение батча для маски
        predicted_mask = predicted_mask_batch[0]
        predicted_label = np.argmax(predicted_label_batch[0])

        # Отображение исходного изображения
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(single_image[0, :, :, 0], cmap='gray')
        plt.title(f'Original (Class: {true_label})')
        plt.axis('off')

        # Отображение истинной маски
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(true_mask[:, :, 0], cmap='gray')
        plt.title('True Mask')
        plt.axis('off')

        # Отображение предсказанной маски
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(predicted_mask[:, :, 0], cmap='gray')
        plt.title(f'Predicted Mask (Class: {predicted_label})')
        plt.axis('off')

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{prefix}test_predictions.png'))
    plt.close()  # Закрываем фигуру
    plt.show()  # Опционально, чтобы отобразить, если запускается интерактивно


# --- Функция для построения и сохранения матрицы ошибок ---
def plot_confusion_matrix(y_true, y_pred, classes, title, filename, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()  # Закрываем фигуру
    plt.show()  # Опционально, чтобы отобразить, если запускается интерактивно
    print(f"Матрица ошибок сохранена в: {os.path.join(save_dir, filename)}")


# --- Функция для сохранения параметров запуска ---
def save_run_parameters(params, save_dir, filename='parameters.txt'):
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        # Удобное форматирование для читаемости
        f.write("--- Run Parameters ---\n")
        f.write(f"Run Timestamp: {params.get('RUN_TIMESTAMP', 'N/A')}\n\n")

        f.write("--- Dataset Information ---\n")
        f.write(f"Dataset Filename: {params.get('DATASET_FILENAME', 'N/A')}\n")
        f.write(f"H5 File Path: {params.get('H5_FILE_PATH', 'N/A')}\n")
        f.write(f"Train Samples: {params.get('TRAIN_SAMPLES', 'N/A')}\n")
        f.write(f"Validation Samples: {params.get('VAL_SAMPLES', 'N/A')}\n")
        f.write(f"Test Samples: {params.get('TEST_SAMPLES', 'N/A')}\n")
        f.write("\n")

        f.write("--- Model Architecture and Parameters ---\n")
        f.write(f"Model Type: U-Net with Classification Head\n")
        f.write(f"Input Shape: {params.get('INPUT_SHAPE', 'N/A')}\n")
        f.write(f"Number of Classes (Classification): {params.get('NUM_CLASSES', 'N/A')}\n")
        f.write(f"Total Model Parameters: {params.get('MODEL_TOTAL_PARAMS', 'N/A')}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Initial Learning Rate: {params.get('LEARNING_RATE', 'N/A')}\n")
        f.write(f"Segmentation Loss Function: Binary Crossentropy\n")
        f.write(f"Classification Loss Function: Categorical Crossentropy\n")
        f.write(f"Metrics: Segmentation Accuracy, Classification Accuracy\n")
        f.write("\n")

        f.write("--- Training Parameters ---\n")
        f.write(f"Epochs: {params.get('EPOCHS', 'N/A')}\n")
        f.write(f"Batch Size: {params.get('BATCH_SIZE', 'N/A')}\n")
        f.write(f"Early Stopping Monitor: {params.get('EARLY_STOPPING_MONITOR', 'N/A')}\n")
        f.write(f"Early Stopping Patience: {params.get('EARLY_STOPPING_PATIENCE', 'N/A')}\n")
        f.write(f"Early Stopping Restore Best Weights: {params.get('EARLY_STOPPING_RESTORE_BEST_WEIGHTS', 'N/A')}\n")
        f.write(f"ReduceLROnPlateau Monitor: {params.get('REDUCE_LR_MONITOR', 'N/A')}\n")
        f.write(f"ReduceLROnPlateau Factor: {params.get('REDUCE_LR_FACTOR', 'N/A')}\n")
        f.write(f"ReduceLROnPlateau Patience: {params.get('REDUCE_LR_PATIENCE', 'N/A')}\n")
        f.write(f"ReduceLROnPlateau Min LR: {params.get('REDUCE_LR_MIN_LR', 'N/A')}\n")
        f.write("\n")

        f.write("--- Final Metrics (Training History) ---\n")
        f.write(f"Final Total Training Loss: {params.get('TOTAL_TRAINING_LOSS_FINAL', 'N/A'):.4f}\n")
        f.write(f"Final Total Validation Loss: {params.get('TOTAL_VALIDATION_LOSS_FINAL', 'N/A'):.4f}\n")
        f.write(
            f"Final Segmentation Training Accuracy: {params.get('SEGMENTATION_TRAINING_ACCURACY_FINAL', 'N/A'):.4f}\n")
        f.write(
            f"Final Segmentation Validation Accuracy: {params.get('SEGMENTATION_VALIDATION_ACCURACY_FINAL', 'N/A'):.4f}\n")
        f.write(
            f"Final Classification Training Accuracy: {params.get('CLASSIFICATION_TRAINING_ACCURACY_FINAL', 'N/A'):.4f}\n")
        f.write(
            f"Final Classification Validation Accuracy: {params.get('CLASSIFICATION_VALIDATION_ACCURACY_FINAL', 'N/A'):.4f}\n")
        f.write("\n")

        f.write("--- Test Metrics (Clean Data) ---\n")
        f.write(f"Test Total Loss: {params.get('TEST_TOTAL_LOSS_CLEAN', 'N/A'):.4f}\n")
        f.write(f"Test Segmentation Loss: {params.get('TEST_SEGMENTATION_LOSS_CLEAN', 'N/A'):.4f}\n")
        f.write(f"Test Classification Loss: {params.get('TEST_CLASSIFICATION_LOSS_CLEAN', 'N/A'):.4f}\n")
        f.write(f"Test Segmentation Accuracy: {params.get('SEGMENTATION_ACCURACY_TEST_CLEAN', 'N/A'):.4f}\n")
        f.write(f"Test Classification Accuracy: {params.get('CLASSIFICATION_ACCURACY_TEST_CLEAN', 'N/A'):.4f}\n")
        f.write("\n")

        f.write(f"Model Save Path: {params.get('MODEL_SAVE_PATH', 'N/A')}\n")
        f.write(f"Results Directory: {save_dir}\n")

    print(f"Параметры запуска сохранены в: {filepath}")


# --- Основная логика ---
if __name__ == '__main__':
    # Создание папок, если их нет
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Добавляем префикс 'unet_' к имени папки, чтобы отличить от других типов моделей
    current_run_dir = os.path.join(RESULTS_FOLDER, f'unet_{run_timestamp}')
    os.makedirs(current_run_dir, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)  # Убедимся, что общая папка для моделей существует
    print(f"Результаты будут сохранены в: {current_run_dir}")

    # Определение пути для сохранения модели
    # Модель сохраняется в общую папку MODELS_FOLDER с префиксом unet_
    MODEL_SAVE_PATH_CURRENT_RUN = os.path.join(MODELS_FOLDER,
                                               f'unet_mnist_segmentation_classification_model_{run_timestamp}.keras')

    # Загрузка данных
    (x_train, mask_train, y_train), \
        (x_val, mask_val, y_val), \
        (x_test, mask_test, y_test) = load_data(H5_FILE_PATH)

    # Создание tf.data.Dataset
    train_ds = create_dataset(x_train, mask_train, y_train, BATCH_SIZE)
    val_ds = create_dataset(x_val, mask_val, y_val, BATCH_SIZE, shuffle=False)
    test_ds = create_dataset(x_test, mask_test, y_test, BATCH_SIZE, shuffle=False)  # Для оценки, без перемешивания

    # Построение модели
    model = build_unet(INPUT_SHAPE, NUM_CLASSES)
    model.summary()

    # Компиляция модели
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss={
                      'segmentation_output': 'binary_crossentropy',
                      'classification_output': 'categorical_crossentropy'
                  },
                  metrics={
                      'segmentation_output': ['accuracy'],
                      'classification_output': ['accuracy']
                  })

    # Обучение модели
    print("\nНачало обучения модели U-Net...")
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        validation_data=val_ds,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor=EARLY_STOPPING_MONITOR,
                                                             patience=EARLY_STOPPING_PATIENCE,
                                                             restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor=REDUCE_LR_MONITOR, factor=REDUCE_LR_FACTOR,
                                                                 patience=REDUCE_LR_PATIENCE, verbose=1,
                                                                 min_lr=REDUCE_LR_MIN_LR)
                        ])
    print("Обучение завершено.")

    # Сохранение модели
    model.save(MODEL_SAVE_PATH_CURRENT_RUN)
    print(f"Модель сохранена в: {MODEL_SAVE_PATH_CURRENT_RUN}")

    # Оценка модели на тестовом наборе (чистые данные)
    print("\nОценка модели на чистом тестовом наборе:")
    # model.evaluate возвращает список потерь и метрик в том порядке, в котором они были указаны в compile
    # Для двух выходов, это будет: [total_loss, seg_loss, cls_loss, seg_accuracy, cls_accuracy]
    results_clean = model.evaluate(test_ds)

    print(f"Общая тестовая потеря (чистые): {results_clean[0]:.4f}")
    print(f"Тестовая потеря сегментации (чистые): {results_clean[1]:.4f}")
    print(f"Тестовая потеря классификации (чистые): {results_clean[2]:.4f}")
    print(f"Тестовая точность сегментации (чистые): {results_clean[3]:.4f}")
    print(f"Тестовая точность классификации (чистые): {results_clean[4]:.4f}")

    # Предсказания для матрицы ошибок (чистые данные)
    # Получаем только классификационные предсказания. model.predict возвращает список массивов,
    # по одному массиву для каждого выхода. В данном случае [predicted_mask_batch, predicted_label_batch]
    _, y_pred_clean_proba = model.predict(x_test, verbose=0)
    y_pred_clean = np.argmax(y_pred_clean_proba, axis=1)
    y_true_clean = np.argmax(y_test, axis=1)  # Истинные метки

    # Построение матрицы ошибок для чистых данных
    class_names = [str(i) for i in range(NUM_CLASSES)]
    plot_confusion_matrix(y_true_clean, y_pred_clean, class_names,
                          'Confusion Matrix (Clean Data)', 'confusion_matrix_clean_cm.png', current_run_dir)

    # Визуализация предсказаний на чистых тестовых примерах
    print("\nВизуализация предсказаний на чистых тестовых примерах:")
    visualize_predictions(model, x_test, mask_test, y_test, num_samples=5, save_dir=current_run_dir, prefix="clean_")

    # Визуализация истории обучения
    plot_training_history(history, current_run_dir)
    print(f"График истории обучения сохранен в: {os.path.join(current_run_dir, 'training_history.png')}")

    # Сохранение параметров запуска в файл
    run_parameters = {
        "RUN_TIMESTAMP": run_timestamp,

        "DATASET_FILENAME": DATASET_FILENAME,
        "H5_FILE_PATH": H5_FILE_PATH,
        "TRAIN_SAMPLES": x_train.shape[0],
        "VAL_SAMPLES": x_val.shape[0],
        "TEST_SAMPLES": x_test.shape[0],

        "INPUT_SHAPE": str(INPUT_SHAPE),
        "NUM_CLASSES": NUM_CLASSES,
        "MODEL_TOTAL_PARAMS": model.count_params(),
        "LEARNING_RATE": LEARNING_RATE,

        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "EARLY_STOPPING_MONITOR": EARLY_STOPPING_MONITOR,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "EARLY_STOPPING_RESTORE_BEST_WEIGHTS": EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
        "REDUCE_LR_MONITOR": REDUCE_LR_MONITOR,
        "REDUCE_LR_FACTOR": REDUCE_LR_FACTOR,
        "REDUCE_LR_PATIENCE": REDUCE_LR_PATIENCE,
        "REDUCE_LR_MIN_LR": REDUCE_LR_MIN_LR,

        "TOTAL_TRAINING_LOSS_FINAL": float(history.history['loss'][-1]),
        "TOTAL_VALIDATION_LOSS_FINAL": float(history.history['val_loss'][-1]),
        "SEGMENTATION_TRAINING_ACCURACY_FINAL": float(history.history['segmentation_output_accuracy'][-1]),
        "SEGMENTATION_VALIDATION_ACCURACY_FINAL": float(history.history['val_segmentation_output_accuracy'][-1]),
        "CLASSIFICATION_TRAINING_ACCURACY_FINAL": float(history.history['classification_output_accuracy'][-1]),
        "CLASSIFICATION_VALIDATION_ACCURACY_FINAL": float(history.history['val_classification_output_accuracy'][-1]),

        "TEST_TOTAL_LOSS_CLEAN": float(results_clean[0]),
        "TEST_SEGMENTATION_LOSS_CLEAN": float(results_clean[1]),
        "TEST_CLASSIFICATION_LOSS_CLEAN": float(results_clean[2]),
        "SEGMENTATION_ACCURACY_TEST_CLEAN": float(results_clean[3]),
        "CLASSIFICATION_ACCURACY_TEST_CLEAN": float(results_clean[4]),

        "MODEL_SAVE_PATH": MODEL_SAVE_PATH_CURRENT_RUN,  # Используем новый путь
        "RESULTS_DIRECTORY": current_run_dir,
    }

    save_run_parameters(run_parameters, current_run_dir)
    print("Все задачи выполнены. Результаты сохранены в папке 'results'.")