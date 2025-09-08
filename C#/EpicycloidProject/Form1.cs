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

        private void buttonDraw_Click(object sender, EventArgs e)
        {
            double a, b;
            if (!double.TryParse(textBoxA.Text, out a) || !double.TryParse(textBoxB.Text, out b))
            {
                MessageBox.Show("Пожалуйста, введите корректные числовые значения для параметров a и b.", "Ошибка ввода", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            Bitmap bmp = new Bitmap(pictureBoxGraph.Width, pictureBoxGraph.Height);
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.White);

                float scale = (float)Math.Min(pictureBoxGraph.Width, pictureBoxGraph.Height) / (2.5f * (float)(a + b));
                int xOffset = pictureBoxGraph.Width / 2;
                int yOffset = pictureBoxGraph.Height / 2;

                Pen axisPen = new Pen(Color.Black, 1);
                g.DrawLine(axisPen, xOffset, 0, xOffset, pictureBoxGraph.Height);
                g.DrawLine(axisPen, 0, yOffset, pictureBoxGraph.Width, yOffset);

                Pen curvePen = new Pen(Color.Blue, 2);
                PointF previousPoint = PointF.Empty;
                int steps = 1000;
                double b_over_a = b / a;

                double t_max = (Math.Abs(b_over_a - Math.Round(b_over_a)) < 0.0001) ? 2 * Math.PI : 2 * Math.PI * 250;

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

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void pictureBoxGraph_Click(object sender, EventArgs e)
        {

        }
    }
}