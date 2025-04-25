import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array

def predict_digit(model_path, image_path):
    """
    Загружает обученную модель Keras и предсказывает цифру на изображении.

    Args:
        model_path (str): Путь к сохраненному файлу модели (.keras).
        image_path (str): Путь к файлу изображения с цифрой.

    Returns:
        tuple: Предсказанная цифра (int) и уверенность предсказания (float),
               или (None, None) в случае ошибки.
    """
    try:
        # Загрузка обученной модели
        model = load_model(model_path)
        print(f"Модель успешно загружена из: {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None, None

    try:
        # Загрузка и предобработка изображения
        img = load_img(image_path, target_size=(28, 28), color_mode='grayscale')
        img_array = img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность пакета
        img_array = np.expand_dims(img_array, axis=-1) # Добавляем размерность канала

        # Выполнение предсказания
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        print(f"Предсказанные вероятности: {predictions}")
        print(f"Предсказанная цифра: {predicted_class}")
        print(f"Уверенность предсказания: {confidence:.4f}")

        return predicted_class, confidence

    except FileNotFoundError:
        print(f"Ошибка: Файл изображения не найден по пути: {image_path}")
        return None, None
    except Exception as e:
        print(f"Произошла ошибка при обработке изображения: {e}")
        return None, None

if __name__ == '__main__':
    # Пример использования функции predict_digit напрямую
    model_path = 'mnist_model/mnist_cnn_model_v4.4.keras'  # Замените на фактический путь к вашей модели
    image_path = 'sample_digit.png'  # Замените на фактический путь к изображению (создайте такой файл для теста)

    predicted_digit_value, confidence = predict_digit(model_path, image_path)

    if predicted_digit_value is not None:
        print(f"При прямом вызове: На изображении, вероятно, изображена цифра: {predicted_digit_value} (Уверенность: {confidence:.2f})")