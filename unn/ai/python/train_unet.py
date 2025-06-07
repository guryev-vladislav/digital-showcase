import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# --- КОНФИГУРАЦИЯ ФАЙЛОВ И ДАТАСЕТА ---
# Папка, где хранятся сгенерированные HDF5 датасеты
DATASET_FOLDER = 'datasets'
# Имя файла HDF5 датасета, который будет использоваться
# ЭТО ИМЯ ДОЛЖНО СОВПАДАТЬ С OUTPUT_H5_FILENAME в generate_augmented_mnist_dataset.py
DATASET_FILENAME = 'synthetic_mnist_large_scale_all_scales.h5'
# Полный путь к файлу датасета
H5_FILE_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

# Папка для сохранения обученных моделей
MODELS_FOLDER = 'models'
# Имя файла, в который будет сохранена обученная модель
MODEL_NAME = 'unet_mnist_segmentation_classification_model.keras'
# Полный путь для сохранения модели
MODEL_SAVE_PATH = os.path.join(MODELS_FOLDER, MODEL_NAME)


# --- КОНФИГУРАЦИЯ МОДЕЛИ И ОБУЧЕНИЯ ---
OUTPUT_IMAGE_SIZE = (112, 112)  # Размер изображений в датасете
NUM_CLASSES = 10  # Количество классов (цифры от 0 до 9)
BATCH_SIZE = 32
EPOCHS = 10 # Вы можете изменить количество эпох


class MnistLargeScaleH5DataLoader:
    def __init__(self, h5_filepath=H5_FILE_PATH, output_image_size=OUTPUT_IMAGE_SIZE, num_classes=NUM_CLASSES):
        """
        Инициализирует загрузчик данных для MNIST Large Scale из HDF5 файла.

        Args:
            h5_filepath (str): Путь к файлу HDF5, содержащему датасет.
            output_image_size (tuple): Ожидаемый размер выходных изображений (ширина, высота).
            num_classes (int): Количество классов для one-hot кодирования меток.
        """
        self.h5_filepath = h5_filepath
        self.OUTPUT_IMAGE_SIZE = output_image_size
        self.NUM_CLASSES = num_classes
        self._load_data_from_h5()

    def _load_data_from_h5(self):
        """
        Загружает изображения, маски и метки из HDF5 файла.
        """
        if not os.path.exists(self.h5_filepath):
            raise FileNotFoundError(f"Файл датасета не найден: {self.h5_filepath}. "
                                    "Пожалуйста, убедитесь, что вы запустили 'generate_augmented_mnist_dataset.py' "
                                    "для создания этого файла (с именем '{os.path.basename(self.h5_filepath)}' в папке '{os.path.dirname(self.h5_filepath)}').")
        print(f"Загрузка данных из HDF5 файла: {self.h5_filepath}...")
        with h5py.File(self.h5_filepath, 'r') as f:
            # Загружаем изображения
            self.x_train = np.array(f["x_train"], dtype=np.float32)
            self.x_val = np.array(f["x_val"], dtype=np.float32)
            self.x_test = np.array(f["x_test"], dtype=np.float32)

            # Загружаем маски (как ground truth для сегментации)
            self.mask_train = np.array(f["mask_train"], dtype=np.float32)
            self.mask_val = np.array(f["mask_val"], dtype=np.float32)
            self.mask_test = np.array(f["mask_test"], dtype=np.float32)

            # Загружаем one-hot закодированные метки классов
            self.y_train = np.array(f["y_train"], dtype=np.float32)  # Они уже one-hot
            self.y_val = np.array(f["y_val"], dtype=np.float32)
            self.y_test = np.array(f["y_test"], dtype=np.float32)

        print("Данные успешно загружены.")
        print(f"Размеры загруженных данных:")
        print(f"x_train: {self.x_train.shape}, mask_train: {self.mask_train.shape}, y_train: {self.y_train.shape}")
        print(f"x_val:   {self.x_val.shape}, mask_val:   {self.mask_val.shape}, y_val:   {self.y_val.shape}")
        print(f"x_test:  {self.x_test.shape}, mask_test:  {self.mask_test.shape}, y_test:  {self.y_test.shape}")

    def get_datasets(self, batch_size=BATCH_SIZE): # Используем константу BATCH_SIZE
        """
        Создает и возвращает tf.data.Dataset для обучения, валидации и тестирования.
        """
        # tf.data.Dataset принимает кортеж (входы, выходы).
        # Для U-Net у нас два выхода: сегментационная маска и метка класса.
        # Поэтому данные должны быть в формате (изображение, (маска, метка_класса))
        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, (self.mask_train, self.y_train)))
        val_ds = tf.data.Dataset.from_tensor_slices((self.x_val, (self.mask_val, self.y_val)))
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, (self.mask_test, self.y_test)))

        # Перемешиваем, батчируем и предварительно загружаем
        train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds


def unet_model(input_size=(OUTPUT_IMAGE_SIZE[0], OUTPUT_IMAGE_SIZE[1], 1), num_classes=NUM_CLASSES):
    """
    Создает базовую архитектуру U-Net для сегментации и классификации.
    Модель имеет два выхода:
    1. Карта сегментации (бинарная, 1 канал)
    2. Метки классов (one-hot, NUM_CLASSES каналов)
    """
    inputs = layers.Input(input_size)

    # --- Кодировщик (Encoder) ---
    # Блок 1
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # 56x56

    # Блок 2
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # 28x28

    # Блок 3
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)  # 14x14

    # Блок 4 (Бутылочное горлышко)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)

    # --- Декодер (Decoder) ---
    # Блок 5 (с использованием skip connection от conv3)
    up6 = layers.concatenate([layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv4), conv3], axis=-1)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)  # 28x28

    # Блок 6 (с использованием skip connection от conv2)
    up7 = layers.concatenate([layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6), conv2], axis=-1)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)  # 56x56

    # Блок 7 (с использованием skip connection от conv1)
    up8 = layers.concatenate([layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv7), conv1], axis=-1)
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv8)  # 112x112

    # --- Выход для сегментации (бинарная маска) ---
    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation_output')(conv8)

    # --- Выход для классификации ---
    # Для классификации берем глобальный пулинг из последнего слоя кодировщика
    # или из слоя после бутылочного горлышка, чтобы получить признаки для классификации.
    # Используем GlobalAveragePooling2D из conv4 для более стабильных признаков.
    flatten_features = layers.GlobalAveragePooling2D()(conv4)
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification_output')(
        flatten_features)

    model = models.Model(inputs=inputs, outputs=[segmentation_output, classification_output])
    return model


def plot_training_history(history):
    """
    Строит графики точности и потерь для обучения и валидации.
    """
    plt.figure(figsize=(12, 5))

    # График потерь
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

    # График точности
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
    plt.show()


# --- Основной скрипт ---
if __name__ == '__main__':
    # Убедимся, что папка для датасетов существует (создается generate_augmented_mnist_dataset.py)
    # Если вы запускаете этот скрипт первым, или переместили датасет,
    # убедитесь, что папка 'datasets' и файл в ней существуют.
    if not os.path.exists(DATASET_FOLDER):
        print(f"Внимание: Папка '{DATASET_FOLDER}' не найдена. "
              "Убедитесь, что вы запустили 'generate_augmented_mnist_dataset.py' для создания датасета.")
        # Можете здесь добавить os.makedirs(DATASET_FOLDER, exist_ok=True)
        # Но лучше, чтобы generate_augmented_mnist_dataset.py был ответственным за создание папки.

    # Создаем папку для сохранения моделей, если ее нет
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    # 1. Загрузка данных
    data_loader = MnistLargeScaleH5DataLoader() # Использует H5_FILE_PATH по умолчанию
    train_ds, val_ds, test_ds = data_loader.get_datasets() # Использует BATCH_SIZE по умолчанию

    # 2. Создание модели U-Net
    # Input shape модели должен соответствовать OUTPUT_IMAGE_SIZE и количеству каналов (1 для серых изображений)
    model = unet_model(input_size=(OUTPUT_IMAGE_SIZE[0], OUTPUT_IMAGE_SIZE[1], 1), num_classes=NUM_CLASSES)
    model.summary()

    # 3. Компиляция модели
    # У U-Net два выхода, поэтому нужны две функции потерь и две метрики.
    # Для сегментации: BinaryCrossentropy, Accuracy
    # Для классификации: CategoricalCrossentropy, Accuracy
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
                        epochs=EPOCHS,  # Используем константу EPOCHS
                        validation_data=val_ds,
                        # Добавим Callbacks для лучшего контроля обучения
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
                        ]
                       )
    print("Обучение завершено.")

    # 5. Сохранение модели
    model.save(MODEL_SAVE_PATH)
    print(f"Модель сохранена в: {MODEL_SAVE_PATH}")


    # 6. Оценка модели на тестовом наборе
    print("\nОценка модели на тестовом наборе:")
    # model.evaluate возвращает потери и метрики в том порядке, в котором они были определены в compile
    # то есть: [total_loss, seg_loss, class_loss, seg_accuracy, class_accuracy]
    results = model.evaluate(test_ds)
    print(f"Общая тестовая потеря: {results[0]:.4f}")
    print(f"Тестовая потеря сегментации: {results[1]:.4f}")
    print(f"Тестовая потеря классификации: {results[2]:.4f}")
    print(f"Тестовая точность сегментации: {results[3]:.4f}")
    print(f"Тестовая точность классификации: {results[4]:.4f}")


    # 7. Построение графиков
    plot_training_history(history)

    # 8. Опционально: Визуализация предсказаний на нескольких тестовых примерах
    print("\nВизуализация предсказаний на тестовых примерах:")
    for images, (true_masks, true_labels) in test_ds.take(1):
        predictions = model.predict(images)
        pred_masks = predictions[0]
        pred_labels = predictions[1]

        plt.figure(figsize=(15, 9))
        for i in range(min(4, images.shape[0])):  # Визуализируем первые 4 примера
            # Оригинальное изображение
            plt.subplot(3, min(4, images.shape[0]), i + 1)
            plt.title(f"Изображение (True: {np.argmax(true_labels[i])})")
            plt.imshow(images[i, :, :, 0].numpy(), cmap='gray')
            plt.axis('off')

            # Истинная маска
            plt.subplot(3, min(4, images.shape[0]), i + 1 + min(4, images.shape[0]))
            plt.title("Истинная маска")
            plt.imshow(true_masks[i, :, :, 0].numpy(), cmap='gray')
            plt.axis('off')

            # Предсказанная маска
            plt.subplot(3, min(4, images.shape[0]), i + 1 + 2 * min(4, images.shape[0]))
            plt.title(f"Предсказанная маска (Pred: {np.argmax(pred_labels[i])})")
            plt.imshow(pred_masks[i, :, :, 0].numpy() > 0.5, cmap='gray')  # Применяем порог 0.5 для бинарной маски
            plt.axis('off')
        plt.tight_layout()
        plt.show()