import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import datetime

# --- КОНФИГУРАЦИЯ ФАЙЛОВ И ДАТАСЕТА ---
DATASET_FOLDER = 'datasets'
DATASET_FILENAME = 'synthetic_mnist_large_scale_flexible_sizes_v1000.h5'
H5_FILE_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

MODELS_FOLDER = 'models'
MODEL_NAME = 'unet_mnist_segmentation_classification_model_v10.keras'  # Обновлена версия модели
MODEL_SAVE_PATH = os.path.join(MODELS_FOLDER, MODEL_NAME)

RESULTS_ROOT_FOLDER = 'results'  # Корневая папка для всех результатов

# --- КОНФИГУРАЦИЯ МОДЕЛИ И ОБУЧЕНИЯ ---
OUTPUT_IMAGE_SIZE = (112, 112)
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 10


class MnistLargeScaleH5DataLoader:
    def __init__(self, h5_filepath=H5_FILE_PATH, output_image_size=OUTPUT_IMAGE_SIZE, num_classes=NUM_CLASSES):
        self.h5_filepath = h5_filepath
        self.OUTPUT_IMAGE_SIZE = output_image_size
        self.NUM_CLASSES = num_classes
        self._load_data_from_h5()

    def _load_data_from_h5(self):
        if not os.path.exists(self.h5_filepath):
            raise FileNotFoundError(f"Файл датасета не найден: {self.h5_filepath}. "
                                    f"Убедитесь, что вы запустили 'generate_augmented_mnist_dataset.py' "
                                    f"для создания этого файла (с именем '{os.path.basename(self.h5_filepath)}' в папке '{os.path.dirname(self.h5_filepath)}').")
        print(f"Загрузка данных из HDF5 файла: {self.h5_filepath}...")

        with h5py.File(self.h5_filepath, 'r') as f:
            num_train_samples = f["x_train"].shape[0]
            num_val_samples = f["x_val"].shape[0]
            num_test_samples = f["x_test"].shape[0]

            self.x_train = np.empty(f["x_train"].shape, dtype=np.float32)
            self.x_val = np.empty(f["x_val"].shape, dtype=np.float32)
            self.x_test = np.empty(f["x_test"].shape, dtype=np.float32)

            self.mask_train = np.empty(f["mask_train"].shape, dtype=np.float32)
            self.mask_val = np.empty(f["mask_val"].shape, dtype=np.float32)
            self.mask_test = np.empty(f["mask_test"].shape, dtype=np.float32)

            self.y_train = np.empty(f["y_train"].shape, dtype=np.float32)
            self.y_val = np.empty(f["y_val"].shape, dtype=np.float32)
            self.y_test = np.empty(f["y_test"].shape, dtype=np.float32)

            print("Загрузка обучающих данных:")
            for i in tqdm(range(num_train_samples), desc="x_train"):
                self.x_train[i] = f["x_train"][i]
            for i in tqdm(range(num_train_samples), desc="mask_train"):
                self.mask_train[i] = f["mask_train"][i]
            for i in tqdm(range(num_train_samples), desc="y_train"):
                self.y_train[i] = f["y_train"][i]

            print("Загрузка валидационных данных:")
            for i in tqdm(range(num_val_samples), desc="x_val"):
                self.x_val[i] = f["x_val"][i]
            for i in tqdm(range(num_val_samples), desc="mask_val"):
                self.mask_val[i] = f["mask_val"][i]
            for i in tqdm(range(num_val_samples), desc="y_val"):
                self.y_val[i] = f["y_val"][i]

            print("Загрузка тестовых данных:")
            for i in tqdm(range(num_test_samples), desc="x_test"):
                self.x_test[i] = f["x_test"][i]
            for i in tqdm(range(num_test_samples), desc="mask_test"):
                self.mask_test[i] = f["mask_test"][i]
            for i in tqdm(range(num_test_samples), desc="y_test"):
                self.y_test[i] = f["y_test"][i]

        print("Данные успешно загружены.")
        print(f"Размеры загруженных данных:")
        print(f"x_train: {self.x_train.shape}, mask_train: {self.mask_train.shape}, y_train: {self.y_train.shape}")
        print(f"x_val:   {self.x_val.shape}, mask_val:   {self.mask_val.shape}, y_val:   {self.y_val.shape}")
        print(f"x_test:  {self.x_test.shape}, mask_test:  {self.mask_test.shape}, y_test:  {self.y_test.shape}")

    def get_datasets(self, batch_size=BATCH_SIZE):
        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, (self.mask_train, self.y_train)))
        val_ds = tf.data.Dataset.from_tensor_slices((self.x_val, (self.mask_val, self.y_val)))
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, (self.mask_test, self.y_test)))

        train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds


def unet_model(input_size=(OUTPUT_IMAGE_SIZE[0], OUTPUT_IMAGE_SIZE[1], 1), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_size)

    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)

    up6 = layers.concatenate([layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv4), conv3], axis=-1)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = layers.concatenate([layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6), conv2], axis=-1)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    up8 = layers.concatenate([layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv7), conv1], axis=-1)
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv8)

    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation_output')(conv8)

    flatten_features = layers.GlobalAveragePooling2D()(conv4)
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification_output')(
        flatten_features)

    model = models.Model(inputs=inputs, outputs=[segmentation_output, classification_output])
    return model


def plot_training_history(history, save_path=None):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Общие потери на обучении')
    plt.plot(history.history['val_loss'], label='Общие потери на валидации')
    plt.plot(history.history['segmentation_output_loss'], label='Потери сегментации на обучении')
    plt.plot(history.history['val_segmentation_output_loss'], label='Потери сегментации на валидации')
    plt.plot(history.history['classification_output_loss'], label='Потери классификации на обучении')
    plt.plot(history.history['val_classification_output_loss'], label='Потери классификации на валидации')
    plt.title('Потери модели')
    plt.ylabel('Потери')
    plt.xlabel('Эпоха')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['segmentation_output_accuracy'], label='Точность сегментации на обучении')
    plt.plot(history.history['val_segmentation_output_accuracy'], label='Точность сегментации на валидации')
    plt.plot(history.history['classification_output_accuracy'], label='Точность классификации на обучении')
    plt.plot(history.history['val_classification_output_accuracy'], label='Точность классификации на валидации')
    plt.title('Точность модели')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"График истории обучения сохранен в: {save_path}")

    plt.show()


def plot_predictions_visualizations(images, true_masks, true_labels, pred_masks, pred_labels, save_path=None):
    plt.figure(figsize=(15, 9))
    for i in range(min(4, images.shape[0])):
        plt.subplot(3, min(4, images.shape[0]), i + 1)
        plt.title(f"Изображение (True: {np.argmax(true_labels[i])})")
        plt.imshow(images[i, :, :, 0].numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(3, min(4, images.shape[0]), i + 1 + min(4, images.shape[0]))
        plt.title("Истинная маска")
        plt.imshow(true_masks[i, :, :, 0].numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(3, min(4, images.shape[0]), i + 1 + 2 * min(4, images.shape[0]))
        plt.title(f"Предсказанная маска (Pred: {np.argmax(pred_labels[i])})")
        plt.imshow(pred_masks[i, :, :, 0].numpy() > 0.5, cmap='gray')
        plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"График визуализации предсказаний сохранен в: {save_path}")

    plt.show()


def save_run_parameters(params, save_folder, timestamp):
    filename = f"run_parameters_{timestamp}.txt"
    filepath = os.path.join(save_folder, filename)

    with open(filepath, 'w') as f:
        f.write("--- Параметры запуска обучения модели U-Net ---\n\n")
        f.write(f"Дата и время запуска: {datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Конфигурация файлов:\n")
        f.write(f"  Папка датасета: {params['DATASET_FOLDER']}\n")
        f.write(f"  Имя файла датасета: {params['DATASET_FILENAME']}\n")
        f.write(f"  Полный путь к датасету: {params['H5_FILE_PATH']}\n")
        f.write(f"  Папка для моделей: {params['MODELS_FOLDER']}\n")
        f.write(f"  Имя файла модели: {params['MODEL_NAME']}\n")
        f.write(f"  Полный путь к модели: {params['MODEL_SAVE_PATH']}\n")
        f.write(f"  Корневая папка для результатов: {params['RESULTS_ROOT_FOLDER']}\n")
        f.write(f"  Папка текущего запуска результатов: {params['CURRENT_RUN_RESULTS_FOLDER']}\n\n")

        f.write("Конфигурация модели и обучения:\n")
        f.write(f"  Размер входного изображения: {params['OUTPUT_IMAGE_SIZE']}\n")
        f.write(f"  Количество классов: {params['NUM_CLASSES']}\n")
        f.write(f"  Размер батча: {params['BATCH_SIZE']}\n")
        f.write(f"  Количество эпох: {params['EPOCHS']}\n\n")

        f.write("Дополнительная информация (может быть добавлена):\n")
        f.write(f"  Оптимизатор: Adam (задан в коде)\n")
        f.write(f"  Функции потерь: BinaryCrossentropy (сегментация), CategoricalCrossentropy (классификация)\n")
        f.write(f"  Метрики: Accuracy (сегментация), Accuracy (классификация)\n")
        f.write(f"  Колбэки: EarlyStopping, ReduceLROnPlateau\n")

    print(f"Параметры запуска сохранены в: {filepath}")


# --- Основной скрипт ---
if __name__ == '__main__':
    # Генерируем метку времени для текущего запуска
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_run_results_folder = os.path.join(RESULTS_ROOT_FOLDER, f"run_{timestamp}")

    # Создаем корневую папку для результатов, если ее нет
    os.makedirs(RESULTS_ROOT_FOLDER, exist_ok=True)
    # Создаем папку для текущего запуска внутри RESULTS_ROOT_FOLDER
    os.makedirs(current_run_results_folder, exist_ok=True)

    # Создаем папку для сохранения моделей, если ее нет
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    # Если папка датасетов не найдена, выводим предупреждение
    if not os.path.exists(DATASET_FOLDER):
        print(f"Внимание: Папка '{DATASET_FOLDER}' не найдена. "
              "Убедитесь, что вы запустили 'generate_augmented_mnist_dataset.py' для создания датасета.")

    # 1. Загрузка данных
    data_loader = MnistLargeScaleH5DataLoader()
    train_ds, val_ds, test_ds = data_loader.get_datasets()

    # 2. Создание модели U-Net
    model = unet_model(input_size=(OUTPUT_IMAGE_SIZE[0], OUTPUT_IMAGE_SIZE[1], 1), num_classes=NUM_CLASSES)
    model.summary()

    # 3. Компиляция модели
    model.compile(optimizer='adam',
                  loss={
                      'segmentation_output': tf.keras.losses.BinaryCrossentropy(),
                      'classification_output': tf.keras.losses.CategoricalCrossentropy()
                  },
                  metrics={
                      'segmentation_output': 'accuracy',
                      'classification_output': 'accuracy'
                  })

    # 4. Обучение модели
    print("\nНачало обучения модели U-Net...")
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        validation_data=val_ds,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,
                                                                 min_lr=1e-6)
                        ]
                        )
    print("Обучение завершено.")

    # 5. Сохранение модели
    # Если вы хотите, чтобы модель сохранялась в папку запуска результатов, измените MODEL_SAVE_PATH здесь.
    # Сейчас она сохраняется в 'models/'.
    model.save(MODEL_SAVE_PATH)
    print(f"Модель сохранена в: {MODEL_SAVE_PATH}")

    # 6. Оценка модели на тестовом наборе
    print("\nОценка модели на тестовом наборе:")
    results_eval = model.evaluate(test_ds)  # Переименовал переменную, чтобы не конфликтовала с RESULTS_ROOT_FOLDER
    print(f"Общая тестовая потеря: {results_eval[0]:.4f}")
    print(f"Тестовая потеря сегментации: {results_eval[1]:.4f}")
    print(f"Тестовая потеря классификации: {results_eval[2]:.4f}")
    print(f"Тестовая точность сегментации: {results_eval[3]:.4f}")
    print(f"Тестовая точность классификации: {results_eval[4]:.4f}")

    # 7. Сохранение и построение графиков
    history_plot_filename = f"training_history.png"  # Имя без timestamp, т.к. он уже в имени папки
    history_plot_path = os.path.join(current_run_results_folder, history_plot_filename)
    plot_training_history(history, save_path=history_plot_path)

    # 8. Визуализация и сохранение предсказаний на нескольких тестовых примерах
    print("\nВизуализация предсказаний на тестовых примерах:")
    images_to_show, true_masks_to_show, true_labels_to_show = next(iter(test_ds.take(1)))
    predictions = model.predict(images_to_show, verbose=0)
    pred_masks = predictions[0]
    pred_labels = predictions[1]

    predictions_plot_filename = f"predictions_visualization.png"  # Имя без timestamp
    predictions_plot_path = os.path.join(current_run_results_folder, predictions_plot_filename)
    plot_predictions_visualizations(images_to_show, true_masks_to_show, true_labels_to_show,
                                    pred_masks, pred_labels, save_path=predictions_plot_path)

    # 9. Сохранение параметров запуска
    run_parameters = {
        'DATASET_FOLDER': DATASET_FOLDER,
        'DATASET_FILENAME': DATASET_FILENAME,
        'H5_FILE_PATH': H5_FILE_PATH,
        'MODELS_FOLDER': MODELS_FOLDER,  # Папка, куда сохраняется модель
        'MODEL_NAME': MODEL_NAME,
        'MODEL_SAVE_PATH': MODEL_SAVE_PATH,  # Полный путь к модели
        'RESULTS_ROOT_FOLDER': RESULTS_ROOT_FOLDER,
        'CURRENT_RUN_RESULTS_FOLDER': current_run_results_folder,  # Новая папка текущего запуска
        'OUTPUT_IMAGE_SIZE': OUTPUT_IMAGE_SIZE,
        'NUM_CLASSES': NUM_CLASSES,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
    }
    save_run_parameters(run_parameters, current_run_results_folder,
                        datetime.datetime.now().timestamp())  # Передаем временную метку как число для datetime.fromtimestamp