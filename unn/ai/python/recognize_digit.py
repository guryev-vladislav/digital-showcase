import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.api.preprocessing import image

def predict_digit(model_path, image_path):
    try:
        # Загрузка модели
        model = keras.models.load_model(model_path)
        print(f"Модель успешно загружена из: {model_path}")

        # Загрузка и предобработка изображения
        img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Добавляем размерность для пакета (batch)

        # Предсказание
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        probability = predictions[0][predicted_class]

        print(f"На изображении, скорее всего, цифра: {predicted_class}")
        print(f"Вероятность: {probability:.4f}")

        # Визуализация вероятностей
        plt.figure(figsize=(10, 5))
        plt.bar(range(10), predictions[0])
        plt.xticks(range(10))
        plt.xlabel("Цифра")
        plt.ylabel("Вероятность")
        plt.title("Вероятности для каждой цифры")
        plt.show()

        # Визуализация входного изображения
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title("Входное изображение")
        plt.show()
    except FileNotFoundError:
        print("Ошибка: Файл модели или изображения не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    model_path = "C:\\NeuralNetworks\\mnist\\mnist_model\\mnist_model_v5.keras"
    image_path = "C:\\NeuralNetworks\\mnist\\example_images\\7.jpg"
    predict_digit(model_path, image_path)