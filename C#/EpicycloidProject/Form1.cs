using System;
using System.Drawing;
using System.Windows.Forms;

namespace EpicycloidProject
{
    public partial class Form1 : Form
    {
        private double scaleFactor = 1.0; // Коэффициент масштаба

        public Form1()
        {
            InitializeComponent();
            trackBarScale.ValueChanged += TrackBarScale_ValueChanged;
        }

        private void TrackBarScale_ValueChanged(object sender, EventArgs e)
        {
            // Обновляем label с текущим масштабом
            labelScale.Text = $"Масштаб: {trackBarScale.Value}%";

            // Сохраняем коэффициент масштаба (делим на 100 для получения множителя)
            scaleFactor = trackBarScale.Value / 100.0;

            // Перерисовываем график если параметры уже введены
            if (!string.IsNullOrEmpty(textBoxA.Text) && !string.IsNullOrEmpty(textBoxB.Text))
            {
                if (double.TryParse(textBoxA.Text, out double a) && double.TryParse(textBoxB.Text, out double b))
                {
                    DrawGraph(a, b);
                }
            }
        }

        private void buttonDraw_Click(object sender, EventArgs e)
        {
            if (!double.TryParse(textBoxA.Text, out double a) || !double.TryParse(textBoxB.Text, out double b))
            {
                MessageBox.Show("Введите корректные числа!");
                return;
            }

            DrawGraph(a, b);
        }

        private void DrawGraph(double a, double b)
        {
            Bitmap bmp = new Bitmap(pictureBoxGraph.Width, pictureBoxGraph.Height);
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.White);
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

                // Используем scaleFactor для масштабирования
                float baseScale = Math.Min(pictureBoxGraph.Width, pictureBoxGraph.Height) / (2.5f * (float)(a + b));
                float actualScale = baseScale * (float)scaleFactor;

                int xOffset = pictureBoxGraph.Width / 2;
                int yOffset = pictureBoxGraph.Height / 2;

                // Рисуем оси
                DrawAxes(g, xOffset, yOffset, actualScale, a + b);

                // Рисуем эпициклоиду
                DrawEpicycloid(g, a, b, actualScale, xOffset, yOffset, Color.Blue);
            }
            pictureBoxGraph.Image = bmp;
        }

        private void DrawAxes(Graphics g, int xOffset, int yOffset, float scale, double radius)
        {
            using (Pen axisPen = new Pen(Color.Gray, 1))
            using (Font font = new Font("Arial", 8))
            using (SolidBrush brush = new SolidBrush(Color.Black))
            {
                g.DrawLine(axisPen, xOffset, 0, xOffset, pictureBoxGraph.Height);
                g.DrawLine(axisPen, 0, yOffset, pictureBoxGraph.Width, yOffset);

                g.DrawString("X", font, brush, pictureBoxGraph.Width - 20, yOffset + 5);
                g.DrawString("Y", font, brush, xOffset + 5, 5);
                g.DrawString("0", font, brush, xOffset + 5, yOffset + 5);
            }
        }

        private void DrawEpicycloid(Graphics g, double a, double b, float scale, int xOffset, int yOffset, Color color)
        {
            using (Pen curvePen = new Pen(color, 2))
            {
                int steps = 1000;
                double t_max = 2 * Math.PI * 2;

                PointF? previousPoint = null;

                for (int i = 0; i <= steps; i++)
                {
                    double t = i * t_max / steps;

                    double x = (a + b) * Math.Cos(t) - a * Math.Cos((a + b) * t / a);
                    double y = (a + b) * Math.Sin(t) - a * Math.Sin((a + b) * t / a);

                    float x_screen = (float)x * scale + xOffset;
                    float y_screen = (float)(-y) * scale + yOffset;

                    PointF currentPoint = new PointF(x_screen, y_screen);

                    if (previousPoint.HasValue)
                    {
                        g.DrawLine(curvePen, previousPoint.Value, currentPoint);
                    }

                    previousPoint = currentPoint;
                }
            }
        }
    }
}