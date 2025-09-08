using System;
using System.Drawing;
using System.Windows.Forms;

namespace EpicycloidProject
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        // Это метод, который будет вызван, когда ты нажмешь на кнопку "Построить график"
        private void buttonDraw_Click(object sender, EventArgs e)
        {
            // 1. Получаем значения a и b из текстовых полей.
            double a, b;
            if (!double.TryParse(textBoxA.Text, out a) || !double.TryParse(textBoxB.Text, out b))
            {
                MessageBox.Show("Пожалуйста, введите корректные числовые значения для параметров a и b.", "Ошибка ввода", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // 2. Создаем "полотно" для рисования.
            Bitmap bmp = new Bitmap(pictureBoxGraph.Width, pictureBoxGraph.Height);
            using (Graphics g = Graphics.FromImage(bmp))
            {
                // Очищаем фон, чтобы каждый раз график рисовался заново.
                g.Clear(Color.White);

                // 3. Настраиваем масштаб и смещение для центрирования графика.
                float scale = (float)Math.Min(pictureBoxGraph.Width, pictureBoxGraph.Height) / (2.5f * (float)(a + b));
                int xOffset = pictureBoxGraph.Width / 2;
                int yOffset = pictureBoxGraph.Height / 2;

                // 4. Рисуем оси X и Y.
                Pen axisPen = new Pen(Color.Black, 1);
                g.DrawLine(axisPen, xOffset, 0, xOffset, pictureBoxGraph.Height);
                g.DrawLine(axisPen, 0, yOffset, pictureBoxGraph.Width, yOffset);

                // Добавляем подписи к осям и единицы измерения
                Font font = new Font("Arial", 10);
                SolidBrush brush = new SolidBrush(Color.Black);

                // Рисуем подписи X и Y
                g.DrawString("X", font, brush, pictureBoxGraph.Width - 20, yOffset + 5);
                g.DrawString("Y", font, brush, xOffset + 5, 5);

                // Добавляем метки в точках пересечения с осями.
                float radius = (float)(a + b);
                float x1_screen = xOffset + radius * scale;
                float x2_screen = xOffset - radius * scale;
                float y1_screen = yOffset - radius * scale;
                float y2_screen = yOffset + radius * scale;

                // Рисуем маленькие кружки в точках пересечения
                int dotSize = 5;
                g.FillEllipse(brush, x1_screen - dotSize / 2, yOffset - dotSize / 2, dotSize, dotSize);
                g.FillEllipse(brush, x2_screen - dotSize / 2, yOffset - dotSize / 2, dotSize, dotSize);
                g.FillEllipse(brush, xOffset - dotSize / 2, y1_screen - dotSize / 2, dotSize, dotSize);
                g.FillEllipse(brush, xOffset - dotSize / 2, y2_screen - dotSize / 2, dotSize, dotSize);

                // Добавляем числовые метки
                g.DrawString("0", font, brush, xOffset - 15, yOffset + 5);
                g.DrawString(radius.ToString("F1"), font, brush, x1_screen + 5, yOffset + 5);
                g.DrawString("-" + radius.ToString("F1"), font, brush, x2_screen - 25, yOffset + 5);
                g.DrawString(radius.ToString("F1"), font, brush, xOffset + 5, y1_screen - 15);
                g.DrawString("-" + radius.ToString("F1"), font, brush, xOffset + 5, y2_screen + 5);


                // 5. Вычисляем и рисуем эпициклоиду.
                Pen curvePen = new Pen(Color.Blue, 2);
                PointF previousPoint = PointF.Empty;
                int steps = 10000;
                double b_over_a = b / a;

                double t_max;
                if (Math.Abs(b_over_a - Math.Round(b_over_a)) < 0.0001)
                {
                    t_max = 2 * Math.PI;
                }
                else
                {
                    long gcd = GCD((long)(b * 1000), (long)(a * 1000));
                    long p = (long)(b * 1000) / gcd;
                    long q = (long)(a * 1000) / gcd;

                    t_max = 2 * Math.PI * q;
                    steps = (int)(steps * q);
                }

                for (int i = 0; i <= steps; i++)
                {
                    double t = i * t_max / steps;

                    double x = (a + b) * Math.Cos(t) - a * Math.Cos((a + b) * t / a);
                    double y = (a + b) * Math.Sin(t) - a * Math.Sin((a + b) * t / a);

                    float x_screen = (float)x * scale + xOffset;
                    float y_screen = (float)y * scale + yOffset;

                    PointF currentPoint = new PointF(x_screen, y_screen);

                    if (i > 0)
                    {
                        g.DrawLine(curvePen, previousPoint, currentPoint);
                    }
                    previousPoint = currentPoint;
                }
            }
            pictureBoxGraph.Image = bmp;
        }

        // Вспомогательный метод для нахождения наибольшего общего делителя (НОД)
        private long GCD(long n1, long n2)
        {
            if (n2 == 0)
            {
                return n1;
            }
            return GCD(n2, n1 % n2);
        }
    }
}