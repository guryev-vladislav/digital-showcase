import os
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np

def save_mnist_samples(save_dir, num_samples=10):
    """
    Загружает случайные изображения из тестового набора MNIST и сохраняет их в указанной папке.

    Args:
        save_dir (str): Путь к папке, куда будут сохранены изображения.
        num_samples (int): Количество изображений для сохранения (по умолчанию 10).
    """
    try:
        (_, _), (test_images, test_labels) = mnist.load_data()
        print("Тестовые данные MNIST успешно загружены.")
    except FileNotFoundError:
        print("Ошибка: Не удалось загрузить данные MNIST.")
        return
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return

    # Создаем папку, если она не существует
    os.makedirs(save_dir, exist_ok=True)
    print(f"Изображения будут сохранены в: {save_dir}")

    # Получаем случайные индексы из тестового набора
    indices = np.random.choice(len(test_images), num_samples, replace=False)

    for i, index in enumerate(indices):
        image = test_images[index]
        label = test_labels[index]
        filename = f"{index}_label_{label}.png"
        filepath = os.path.join(save_dir, filename)

        # Сохраняем изображение
        plt.imsave(filepath, image, cmap='gray')
        print(f"Сохранено: {filename}")

    print("Сохранение образцов завершено.")

if __name__ == '__main__':
    save_path = 'mnist_samples'  # Название папки по умолчанию
    save_mnist_samples(save_path, num_samples=40)