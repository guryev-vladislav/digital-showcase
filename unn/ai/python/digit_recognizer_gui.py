import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import predict_digit_backend as backend  # Импортируем бэкенд

class DigitRecognizerApp:
    def __init__(self, master, model_path, example_dir):
        self.master = master
        master.title("Распознавание цифр")

        # Устанавливаем начальный размер окна
        master.geometry("400x450")

        self.model_path = model_path  # Получаем путь к модели извне
        self.example_dir = example_dir # Получаем путь к папке с примерами
        self.image_path = ""
        self.img_label = tk.Label(master)
        self.img_label.pack(pady=20)  # Увеличиваем вертикальный отступ

        self.load_button = tk.Button(master, text="Загрузить изображение", command=self.load_image, padx=20, pady=10)
        self.load_button.pack(pady=10)  # Увеличиваем вертикальный отступ

        self.load_example_button = tk.Button(master, text="Выбрать из примеров", command=self.load_example_image, padx=20, pady=10)
        self.load_example_button.pack(pady=10)  # Увеличиваем вертикальный отступ

        self.predict_button = tk.Button(master, text="Распознать цифру", command=self.predict, state=tk.DISABLED, padx=20, pady=10)
        self.predict_button.pack(pady=10)  # Увеличиваем вертикальный отступ

        self.result_label = tk.Label(master, text="Результат: ", font=("Arial", 16)) # Увеличиваем шрифт результата
        self.result_label.pack(pady=20)  # Увеличиваем вертикальный отступ

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            initialdir=".",
            title="Выберите изображение",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*"))
        )
        self._display_selected_image()

    def load_example_image(self):
        self.image_path = filedialog.askopenfilename(
            initialdir=self.example_dir,
            title="Выберите изображение из примеров",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*"))
        )
        self._display_selected_image()

    def _display_selected_image(self):
        if self.image_path:
            try:
                img = Image.open(self.image_path).resize((128, 128))
                img = ImageTk.PhotoImage(img)
                self.img_label.config(image=img)
                self.img_label.image = img  # keep a reference!
                self.predict_button.config(state=tk.NORMAL)
                self.result_label.config(text="Результат: ") # Сброс предыдущего результата
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
                self.image_path = ""
                self.predict_button.config(state=tk.DISABLED)

    def predict(self):
        if not self.image_path:
            return

        predicted_class, confidence = backend.predict_digit(self.model_path, self.image_path)

        if predicted_class is not None:
            self.result_label.config(text=f"Результат: {predicted_class} (Уверенность: {confidence:.2f})")
        else:
            messagebox.showerror("Ошибка", "Не удалось выполнить распознавание.")

if __name__ == "__main__":
    model_path = 'mnist_model/mnist_cnn_model_v5.0.keras'  # Определяем путь к модели здесь
    example_dir_path = os.path.join(os.path.dirname(__file__), "mnist_samples") # Определяем путь к папке с примерами
    print(f"Путь к папке с примерами изображений: {example_dir_path}")

    root = tk.Tk()
    app = DigitRecognizerApp(root, model_path, example_dir_path)  # Передаем путь к папке с примерами
    root.mainloop()