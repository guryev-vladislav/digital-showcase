import tensorflow as tf
import numpy as np
import h5py
import os
from tqdm import tqdm # Добавляем tqdm для прогресс-баров

# --- Конфигурация датасета ---
OUTPUT_IMAGE_SIZE = (112, 112)  # Фиксированный размер выходного изображения
NUM_CLASSES = 10  # Цифры от 0 до 9

# Множители масштаба для аугментации.
AUGMENTATION_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0)

# --- КОНСТАНТЫ ДЛЯ УПРАВЛЕНИЯ РАЗМЕРОМ ДАТАСЕТА ---
# ОБЩЕЕ количество ИСХОДНЫХ ТРЕНИРОВОЧНЫХ изображений MNIST для использования.
# Оригинальный MNIST имеет 60,000 тренировочных изображений.
# Если вы хотите использовать меньшее подмножество, измените эту константу.
ORIGINAL_MNIST_TRAIN_SAMPLES_TO_USE = 20000

# Процент от ORIGINAL_MNIST_TRAIN_SAMPLES_TO_USE, который пойдет на валидацию.
VALIDATION_SPLIT_RATIO = 0.1  # 10% для валидации, 90% для тренировки

# ОБЩЕЕ количество ИСХОДНЫХ ТЕСТОВЫХ изображений MNIST для использования.
# Оригинальный MNIST имеет 10,000 тестовых изображений.
ORIGINAL_MNIST_TEST_SAMPLES_TO_USE = 10000

# --- Конфигурация папок и имен файлов ---
DATASET_FOLDER = 'datasets' # Новая константа для папки датасета
OUTPUT_H5_FILENAME_ONLY = 'synthetic_mnist_large_scale_flexible_sizes.h5' # Только имя файла
OUTPUT_H5_FULL_PATH = os.path.join(DATASET_FOLDER, OUTPUT_H5_FILENAME_ONLY) # Полный путь


def _augment_image_and_create_mask(image, label, scale_factor):
    """
    Применяет заданный масштаб к изображению, затем "встраивает" его в центр
    холста OUTPUT_IMAGE_SIZE. Создает бинарную маску и one-hot метку.

    Args:
        image (tf.Tensor): Исходное изображение.
        label (tf.Tensor): Исходная метка.
        scale_factor (float): Множитель масштаба для применения.

    Returns:
        tuple: (финальное_изображение, маска, one_hot_метка)
    """
    original_h, original_w = tf.shape(image)[0], tf.shape(image)[1]
    new_h = tf.cast(tf.cast(original_h, tf.float32) * scale_factor, tf.int32)
    new_w = tf.cast(tf.cast(original_w, tf.float32) * scale_factor, tf.int32)

    scaled_image = tf.image.resize(image, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)

    start_h = tf.cast((OUTPUT_IMAGE_SIZE[0] - new_h) / 2, tf.int32)
    start_w = tf.cast((OUTPUT_IMAGE_SIZE[1] - new_w) / 2, tf.int32)

    pad_top = tf.maximum(0, start_h)
    pad_bottom = tf.maximum(0, OUTPUT_IMAGE_SIZE[0] - (start_h + new_h))
    pad_left = tf.maximum(0, start_w)
    pad_right = tf.maximum(0, OUTPUT_IMAGE_SIZE[1] - (start_w + new_w))

    padded_scaled_image = tf.pad(scaled_image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    crop_start_h = tf.maximum(0, -start_h)
    crop_start_w = tf.maximum(0, -start_w)

    final_image = tf.image.crop_to_bounding_box(
        padded_scaled_image,
        offset_height=crop_start_h,
        offset_width=crop_start_w,
        target_height=OUTPUT_IMAGE_SIZE[0],
        target_width=OUTPUT_IMAGE_SIZE[1]
    )

    mask = tf.where(final_image > 0.1, 1.0, 0.0)

    one_hot_label = tf.one_hot(label, depth=NUM_CLASSES)

    return final_image, mask, one_hot_label


def generate_and_save_augmented_mnist(output_full_path=OUTPUT_H5_FULL_PATH):
    """
    Генерирует аугментированный датасет MNIST, где каждое исходное изображение
    аугментируется со всеми заданными масштабами, и сохраняет его в HDF5 файл.
    Размеры наборов данных зависят от констант в начале файла.
    """
    print("Загрузка оригинального датасета MNIST...")
    (x_train_original, y_train_original), \
        (x_test_original, y_test_original) = tf.keras.datasets.mnist.load_data()
    print("Оригинальный MNIST загружен.")

    x_train_original = np.expand_dims(x_train_original, axis=-1)
    x_test_original = np.expand_dims(x_test_original, axis=-1)

    x_train_original = x_train_original.astype('float32') / 255.0
    x_test_original = x_test_original.astype('float32') / 255.0

    actual_train_samples = min(ORIGINAL_MNIST_TRAIN_SAMPLES_TO_USE, len(x_train_original))
    actual_test_samples = min(ORIGINAL_MNIST_TEST_SAMPLES_TO_USE, len(x_test_original))

    num_train_for_split = int(actual_train_samples * (1 - VALIDATION_SPLIT_RATIO))
    num_val_for_split = actual_train_samples - num_train_for_split

    x_train_split = x_train_original[:num_train_for_split]
    y_train_split = y_train_original[:num_train_for_split]

    x_val_split = x_train_original[num_train_for_split:num_train_for_split + num_val_for_split]
    y_val_split = y_train_original[num_train_for_split:num_train_for_split + num_val_for_split]

    x_test_split = x_test_original[:actual_test_samples]
    y_test_split = y_test_original[:actual_test_samples]

    print(f"\nБудет сгенерировано аугментированных данных:")
    print(
        f"Тренировочный набор: {len(x_train_split)} исходных изображений * {len(AUGMENTATION_FACTORS)} масштабов = {len(x_train_split) * len(AUGMENTATION_FACTORS)} аугментированных изображений.")
    print(
        f"Валидационный набор: {len(x_val_split)} исходных изображений * {len(AUGMENTATION_FACTORS)} масштабов = {len(x_val_split) * len(AUGMENTATION_FACTORS)} аугментированных изображений.")
    print(
        f"Тестовый набор: {len(x_test_split)} исходных изображений * {len(AUGMENTATION_FACTORS)} масштабов = {len(x_test_split) * len(AUGMENTATION_FACTORS)} аугментированных изображений.")

    print("\nПрименение аугментации для всех масштабов к каждому изображению (это может занять значительное время)...")

    all_x_train, all_mask_train, all_y_train = [], [], []
    all_x_val, all_mask_val, all_y_val = [], [], []
    all_x_test, all_mask_test, all_y_test = [], [], []

    # Обработка тренировочного набора с прогресс-баром
    print(f"Обработка тренировочного набора ({len(x_train_split)} исходных изображений):")
    for i in tqdm(range(len(x_train_split)), desc="Генерация train data"):
        original_img = x_train_split[i]
        original_lbl = y_train_split[i]
        for scale_factor in AUGMENTATION_FACTORS:
            img_tensor = tf.constant(original_img, dtype=tf.float32)
            lbl_tensor = tf.constant(original_lbl, dtype=tf.int32)
            scale_tensor = tf.constant(scale_factor, dtype=tf.float32)
            augmented_img, mask, one_hot_lbl = _augment_image_and_create_mask(img_tensor, lbl_tensor, scale_tensor)

            all_x_train.append(augmented_img.numpy())
            all_mask_train.append(mask.numpy())
            all_y_train.append(one_hot_lbl.numpy())

    # Обработка валидационного набора с прогресс-баром
    print(f"Обработка валидационного набора ({len(x_val_split)} исходных изображений):")
    for i in tqdm(range(len(x_val_split)), desc="Генерация val data"):
        original_img = x_val_split[i]
        original_lbl = y_val_split[i]
        for scale_factor in AUGMENTATION_FACTORS:
            img_tensor = tf.constant(original_img, dtype=tf.float32)
            lbl_tensor = tf.constant(original_lbl, dtype=tf.int32)
            scale_tensor = tf.constant(scale_factor, dtype=tf.float32)
            augmented_img, mask, one_hot_lbl = _augment_image_and_create_mask(img_tensor, lbl_tensor, scale_tensor)

            all_x_val.append(augmented_img.numpy())
            all_mask_val.append(mask.numpy())
            all_y_val.append(one_hot_lbl.numpy())

    # Обработка тестового набора с прогресс-баром
    print(f"Обработка тестового набора ({len(x_test_split)} исходных изображений):")
    for i in tqdm(range(len(x_test_split)), desc="Генерация test data"):
        original_img = x_test_split[i]
        original_lbl = y_test_split[i]
        for scale_factor in AUGMENTATION_FACTORS:
            img_tensor = tf.constant(original_img, dtype=tf.float32)
            lbl_tensor = tf.constant(original_lbl, dtype=tf.int32)
            scale_tensor = tf.constant(scale_factor, dtype=tf.float32)
            augmented_img, mask, one_hot_lbl = _augment_image_and_create_mask(img_tensor, lbl_tensor, scale_tensor)

            all_x_test.append(augmented_img.numpy())
            all_mask_test.append(mask.numpy())
            all_y_test.append(one_hot_lbl.numpy())

    x_train_final = np.array(all_x_train, dtype=np.float32)
    mask_train_final = np.array(all_mask_train, dtype=np.float32)
    y_train_final = np.array(all_y_train, dtype=np.float32)

    x_val_final = np.array(all_x_val, dtype=np.float32)
    mask_val_final = np.array(all_mask_val, dtype=np.float32)
    y_val_final = np.array(all_y_val, dtype=np.float32)

    x_test_final = np.array(all_x_test, dtype=np.float32)
    mask_test_final = np.array(all_mask_test, dtype=np.float32)
    y_test_final = np.array(all_y_test, dtype=np.float32)

    # --- Создание папки, если её нет ---
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    print(f"Папка для датасета '{DATASET_FOLDER}' проверена/создана.")

    print(f"Сохранение аугментированного датасета в HDF5 файл: {output_full_path}...")
    with h5py.File(output_full_path, 'w') as f:
        f.create_dataset('x_train', data=x_train_final, compression="gzip", compression_opts=9)
        f.create_dataset('y_train', data=y_train_final, compression="gzip", compression_opts=9)
        f.create_dataset('mask_train', data=mask_train_final, compression="gzip", compression_opts=9)

        f.create_dataset('x_val', data=x_val_final, compression="gzip", compression_opts=9)
        f.create_dataset('y_val', data=y_val_final, compression="gzip", compression_opts=9)
        f.create_dataset('mask_val', data=mask_val_final, compression="gzip", compression_opts=9)

        f.create_dataset('x_test', data=x_test_final, compression="gzip", compression_opts=9)
        f.create_dataset('y_test', data=y_test_final, compression="gzip", compression_opts=9)
        f.create_dataset('mask_test', data=mask_test_final, compression="gzip", compression_opts=9)

    print(f"Датасет успешно сгенерирован и сохранен в '{output_full_path}'")
    print(f"Размеры сохраненных данных:")
    print(f"x_train: {x_train_final.shape}, y_train: {y_train_final.shape}, mask_train: {mask_train_final.shape}")
    print(f"x_val:   {x_val_final.shape}, y_val:   {y_val_final.shape}, mask_val:   {mask_val_final.shape}")
    print(f"x_test:  {x_test_final.shape}, y_test:  {y_test_final.shape}, mask_test:  {mask_test_final.shape}")


if __name__ == '__main__':
    generate_and_save_augmented_mnist()

    # --- Проверка сгенерированного файла ---
    print(f"\nПроверка загрузки сгенерированного файла '{OUTPUT_H5_FULL_PATH}'...")
    import matplotlib.pyplot as plt

    with h5py.File(OUTPUT_H5_FULL_PATH, 'r') as f: # Изменено на OUTPUT_H5_FULL_PATH
        x_train_loaded = np.array(f['x_train'])
        y_train_loaded = np.array(f['y_train'])
        mask_train_loaded = np.array(f['mask_train'])

        print(f"x_train загружено: {x_train_loaded.shape}")
        print(f"y_train загружено: {y_train_loaded.shape}")
        print(f"mask_train загружено: {mask_train_loaded.shape}")

        num_samples_to_plot = 5
        plt.figure(figsize=(15, 6))
        for i in range(num_samples_to_plot):
            idx = np.random.randint(0, len(x_train_loaded))

            plt.subplot(2, num_samples_to_plot, i + 1)
            plt.title(f"Изображение (класс: {np.argmax(y_train_loaded[idx])})")
            plt.imshow(x_train_loaded[idx, :, :, 0], cmap='gray')
            plt.axis('off')

            plt.subplot(2, num_samples_to_plot, i + 1 + num_samples_to_plot)
            plt.title("Маска")
            plt.imshow(mask_train_loaded[idx, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()