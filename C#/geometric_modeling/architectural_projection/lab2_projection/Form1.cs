using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace lab2_projection
{
    public partial class Form1 : Form
    {
        Bitmap bmp;
        Graphics g;
        Pen pen0 = new Pen(Color.Black, 1);
        Pen penx = new Pen(Color.Red, 2);
        Pen peny = new Pen(Color.Blue, 2);
        Pen penz = new Pen(Color.Lime, 2);

        Pen penx2 = new Pen(Color.Red, 2);
        Pen peny2 = new Pen(Color.Blue, 2);
        Pen penz2 = new Pen(Color.Lime, 2);


        List<Pen> pens;
        matrix Rx_plus, Ry_plus, Rz_plus, Rx_minus, Ry_minus, Rz_minus, Ax, Az,Ay, A_iso;
        mesh axe_x, axe_y, axe_z, floor1, floor2, floor2_1, floor3, col1, col2, roof, wall1, wall2;
        List<mesh> my_object0, my_object;

        private void button4_Click(object sender, EventArgs e)
        {
            float zoom = 1.1F;
            mult_all(zooming(zoom, zoom, zoom), ref my_object);
            draw_pic();
        }

        private void button5_Click(object sender, EventArgs e)
        {
            float zoom = 0.9F;
            mult_all(zooming(zoom, zoom, zoom), ref my_object);
            draw_pic();
        }

        private void button7_Click(object sender, EventArgs e)
        {
            float move = 15F;
            mult_all(moving_on(move, 0, 0), ref my_object);
            draw_pic();
        }

        private void button6_Click(object sender, EventArgs e)
        {
            float move = -15F;
            mult_all(moving_on(move, 0, 0), ref my_object);
            draw_pic();
        }

        private void button11_Click(object sender, EventArgs e)
        {
            float move = 15F;
            mult_all(moving_on(0, 0, move), ref my_object);
            draw_pic();
        }

        private void button10_Click(object sender, EventArgs e)
        {
            float move = -15F;
            mult_all(moving_on(0, 0, move), ref my_object);
            draw_pic();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            
        }



        bool is_down = false;
        int cursor_x, cursor_y;

        public Form1()
        {
            InitializeComponent();

            penx.DashStyle = DashStyle.Custom;
            penx.DashPattern = new float[] { 5, 10 };

            peny.DashStyle = DashStyle.Custom;
            peny.DashPattern = new float[] { 5, 10 };

            penz.DashStyle = DashStyle.Custom;
            penz.DashPattern = new float[] { 5, 10 };


            bmp = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            g = Graphics.FromImage(bmp);
            g.Clear(Color.White);
            g.TranslateTransform(pictureBox1.Width / 2 - 50, pictureBox1.Height / 2 + 100);
            g.ScaleTransform(1, -1);

            Rx_plus = new matrix(Rx((float)0.1));
            Rx_minus = new matrix(Rx(-(float)0.1));
            Ry_plus = new matrix(Ry((float)0.1));
            Ry_minus = new matrix(Ry(-(float)0.1));
            Rz_plus = new matrix(Rz((float)0.1));
            Rz_minus = new matrix(Rz(-(float)0.1));

            Ax = new matrix();
            Ax.Add(0, 1, 0, 0);
            Ax.Add(1, 0, 0, 0);
            Ax.Add(0, 0, 1, 0);
            Ax.Add(0, 0, 0, 1);

            Ay = new matrix();
            // cos(-135°) = -√2/2 ≈ -0.7071, sin(-135°) = -√2/2 ≈ -0.7071
            // cos(-45°) = √2/2 ≈ 0.7071, sin(-45°) = -√2/2 ≈ -0.7071

            Ay.Add((float)-0.7071, (float)0.5000, (float)-0.5000, 0);        // X компонента
            Ay.Add((float)-0.7071, (float)-0.5000, (float)0.5000, 0);        // Y компонента  
            Ay.Add((float)0.0000, (float)0.7071, (float)0.7071, 0);          // Z компонента
            Ay.Add(0, 0, 0, 1);                 // W компонента

            Az = new matrix();
            Az.Add(1, 0, 0, 0);
            Az.Add(0, 0, 1, 0);
            Az.Add(0, 1, 0, 0);
            Az.Add(0, 0, 0, 1);

            A_iso = new matrix();
            A_iso.Add(1, 0, 0, 0);
            A_iso.Add(0, 0, 1, 0);
            A_iso.Add(0, 1, 0, 0);
            A_iso.Add(0, 0, 0, 1);

            init_obj();
            draw_pic();
        }
        public matrix moving_on(float dx, float dy, float dz)
        {
            matrix A = new matrix();
            A.Add(1, 0, 0, 0);
            A.Add(0, 1, 0, 0);
            A.Add(0, 0, 1, 0);
            A.Add(dx, dy, dz, 1);
            return A;
        }
        public matrix zooming(float kx, float ky, float kz)
        {
            matrix A = new matrix();
            A.Add(kx, 0, 0, 0);
            A.Add(0, ky, 0, 0);
            A.Add(0, 0, kz, 0);
            A.Add(0, 0, 0, 1);
            return A;
        }
        public matrix Rx(float fi)
        {
            matrix res = new matrix();
            res.Add(1, 0, 0, 0);
            res.Add(0, (float)Math.Cos(fi), -(float)Math.Sin(fi), 0);
            res.Add(0, (float)Math.Sin(fi), (float)Math.Cos(fi), 0);
            res.Add(0, 0, 0, 1);
            return res;
        }
        public matrix Ry(float fi)
        {
            matrix res = new matrix();
            res.Add((float)Math.Cos(fi), 0, (float)Math.Sin(fi), 0);
            res.Add(0, 1, 0, 0);
            res.Add(-(float)Math.Sin(fi), 0, (float)Math.Cos(fi), 0);
            res.Add(0, 0, 0, 0);
            return res;
        }
        public matrix Rz(float fi)
        {
            matrix res = new matrix();
            res.Add((float)Math.Cos(fi), (float)Math.Sin(fi), 0, 0);
            res.Add(-(float)Math.Sin(fi), (float)Math.Cos(fi), 0, 0);
            res.Add(0, 0, 1, 0);
            res.Add(0, 0, 0, 1);
            return res;
        }
        public void init_obj()
        {
            my_object0 = new List<mesh>();
            my_object = new List<mesh>();
            pens = new List<Pen>();



            roof = new mesh();
            roof.vertices.Add(0, 0, 0, 1); //0
            roof.edges.Add(new int[] { 1, 3 });
            roof.vertices.Add(0, 18, 0, 1); //1
            roof.edges.Add(new int[] { 2 });
            roof.vertices.Add(10, 18, 0, 1); //2
            roof.edges.Add(new int[] { 3 });
            roof.vertices.Add(10, 0, 0, 1); //3
            roof.edges.Add(new int[] { 0 }); // Замкнули верхний прямоугольник
            roof.vertices.Add(0, 0, 1, 1); //4 (Новая вершина для толщины)
            roof.edges.Add(new int[] { 5, 7, 0 }); // Связь с новыми вершинами для толщины
            roof.vertices.Add(0, 18, 1, 1); //5
            roof.edges.Add(new int[] { 6, 1 });
            roof.vertices.Add(10, 18, 1, 1); //6
            roof.edges.Add(new int[] { 7, 2 });
            roof.vertices.Add(10, 0, 1, 1); //7
            roof.edges.Add(new int[] { 4, 3 }); // Замкнули нижний прямоугольник
            roof.vertices *= moving_on(0, 0, 18); // Переместили крышу наверх
            my_object0.Add(roof);
            pens.Add(pen0);

            wall1 = new mesh();
            wall1.vertices.Add(0, 0, 0, 1); //0
            wall1.edges.Add(new int[] { 1, 7 });
            wall1.vertices.Add(0, 16, 0, 1); //1
            wall1.edges.Add(new int[] { 2, 9 });
            wall1.vertices.Add(6, 16, 0, 1); //2
            wall1.edges.Add(new int[] { 3, 10 });
            wall1.vertices.Add(6, 11, 0, 1); //3 // Верхняя правая часть проема
            wall1.edges.Add(new int[] { 4 });
            wall1.vertices.Add(3, 11, 0, 1); //4 // Верхняя левая часть проема
            wall1.edges.Add(new int[] { 5 });
            wall1.vertices.Add(3, 5, 0, 1); //5 // Нижняя левая часть проема
            wall1.edges.Add(new int[] { 6 });
            wall1.vertices.Add(6, 5, 0, 1); //6 // Нижняя правая часть проема
            wall1.edges.Add(new int[] { 7 }); // Связь 6 с 7
            wall1.vertices.Add(6, 0, 0, 1); //7
            wall1.edges.Add(new int[] { 0 }); // Связь 7 с 0

            wall1.vertices.Add(0, 0, 16, 1); //8 (Задняя часть стены)
            wall1.edges.Add(new int[] { 9, 11 }); // Связь 8 с 9 и 11
            wall1.vertices.Add(0, 16, 16, 1); //9
            wall1.edges.Add(new int[] { 10 }); // Связь 9 с 10
            wall1.vertices.Add(6, 16, 16, 1);//10
            wall1.edges.Add(new int[] { 11 }); // Связь 10 с 11
            wall1.vertices.Add(6, 0, 16, 1); //11
            wall1.edges.Add(new int[] { 8 }); // Связь 11 с 8 (замыкаем заднюю часть)

            // Добавляем связи между передней и задней частью
            wall1.edges[0] = new int[] { 1, 7, 8 }; // Вершина 0 с 1, 7, 8
            wall1.edges[1] = new int[] { 2, 9 }; // Вершина 1 с 2, 9
            wall1.edges[2] = new int[] { 3, 10 }; // Вершина 2 с 3, 10
            wall1.edges[3] = new int[] { 4 }; // Вершина 3 с 4
            wall1.edges[4] = new int[] { 5 }; // Вершина 4 с 5
            wall1.edges[5] = new int[] { 6 }; // Вершина 5 с 6
            wall1.edges[6] = new int[] { 7 }; // Вершина 6 с 7
            wall1.edges[7] = new int[] { 0, 11 }; // Вершина 7 с 0, 11
            wall1.edges[8] = new int[] { 9 }; // Вершина 8 с 9
            wall1.edges[9] = new int[] { 10 }; // Вершина 9 с 10
            wall1.edges[10] = new int[] { 11 }; // Вершина 10 с 11
            wall1.edges[11] = new int[] { 8 }; // Вершина 11 с 8

            wall1.vertices *= moving_on(1, 1, 2);
            my_object0.Add(wall1);
            pens.Add(pen0);

            wall2 = new mesh();
            wall2.vertices.Add(0, 1, 0, 1); //0
            wall2.edges.Add(new int[] { 1, 7 });
            wall2.vertices.Add(0, 15, 0, 1); //1
            wall2.edges.Add(new int[] { 2, 9 });
            wall2.vertices.Add(5, 15, 0, 1); //2
            wall2.edges.Add(new int[] { 3, 10 });
            wall2.vertices.Add(5, 11, 0, 1); //3
            wall2.edges.Add(new int[] { 4 });
            wall2.vertices.Add(4, 11, 0, 1); //4
            wall2.edges.Add(new int[] { 5 });
            wall2.vertices.Add(4, 5, 0, 1); //5
            wall2.edges.Add(new int[] { 6 });
            wall2.vertices.Add(5, 5, 0, 1); //6
            wall2.edges.Add(new int[] { 7 });
            wall2.vertices.Add(5, 1, 0, 1); //7
            wall2.edges.Add(new int[] { 0 }); // Связь 7 с 0

            wall2.vertices.Add(0, 1, 16, 1); //8 (Задняя часть стены)
            wall2.edges.Add(new int[] { 9, 11 }); // Связь 8 с 9 и 11
            wall2.vertices.Add(0, 15, 16, 1); //9
            wall2.edges.Add(new int[] { 10 }); // Связь 9 с 10
            wall2.vertices.Add(5, 15, 16, 1);//10
            wall2.edges.Add(new int[] { 11 }); // Связь 10 с 11
            wall2.vertices.Add(5, 1, 16, 1); //11
            wall2.edges.Add(new int[] { 8 }); // Связь 11 с 8 (замыкаем заднюю часть)

            // Добавляем связи между передней и задней частью
            wall2.edges[0] = new int[] { 1, 7, 8 };
            wall2.edges[1] = new int[] { 2, 9 };
            wall2.edges[2] = new int[] { 3, 10 };
            wall2.edges[3] = new int[] { 4 };
            wall2.edges[4] = new int[] { 5 };
            wall2.edges[5] = new int[] { 6 };
            wall2.edges[6] = new int[] { 7 };
            wall2.edges[7] = new int[] { 0, 11 };
            wall2.edges[8] = new int[] { 9 };
            wall2.edges[9] = new int[] { 10 };
            wall2.edges[10] = new int[] { 11 };
            wall2.edges[11] = new int[] { 8 };

            wall2.vertices *= moving_on(1, 1, 2);
            my_object0.Add(wall2);
            pens.Add(pen0);

            floor1 = new mesh();
            floor1.vertices.Add(0, 0, 0, 1); //0
            floor1.edges.Add(new int[] { 1, 3, 4 });
            floor1.vertices.Add(0, 18, 0, 1); //1
            floor1.edges.Add(new int[] { 2, 5 });
            floor1.vertices.Add(8, 18, 0, 1); //2
            floor1.edges.Add(new int[] { 3, 6 });
            floor1.vertices.Add(8, 0, 0, 1); //3
            floor1.edges.Add(new int[] { 7 });
            floor1.vertices.Add(0, 0, 2, 1); //4
            floor1.edges.Add(new int[] { 5, 7 });
            floor1.vertices.Add(0, 18, 2, 1); //5
            floor1.edges.Add(new int[] { 6 });
            floor1.vertices.Add(8, 18, 2, 1); //6
            floor1.edges.Add(new int[] { 7 });
            floor1.vertices.Add(8, 0, 2, 1); //7
            floor1.edges.Add(null);
            my_object0.Add(floor1);
            pens.Add(pen0);

            floor3 = new mesh();
            floor3.vertices.Add(0, 0, 2, 1); //0
            floor3.edges.Add(new int[] { 1, 9 });
            floor3.vertices.Add(0, 8, 2, 1); //1
            floor3.edges.Add(new int[] { 2 });
            floor3.vertices.Add(2, 8, 2, 1); //2
            floor3.edges.Add(new int[] { 3, 9 });
            floor3.vertices.Add(2, 8, 1, 1); //3
            floor3.edges.Add(new int[] { 4, 8 });
            floor3.vertices.Add(3, 8, 1, 1); //4
            floor3.edges.Add(new int[] { 5, 7 });
            floor3.vertices.Add(3, 8, 0, 1); //5
            floor3.edges.Add(new int[] { 6 });
            floor3.vertices.Add(3, 0, 0, 1); //6
            floor3.edges.Add(new int[] { 7 });
            floor3.vertices.Add(3, 0, 1, 1); //7
            floor3.edges.Add(new int[] { 8 });
            floor3.vertices.Add(2, 0, 1, 1); //8
            floor3.edges.Add(new int[] { 9 });
            floor3.vertices.Add(2, 0, 2, 1); //9
            floor3.edges.Add(null);
            floor3.vertices *= moving_on(7, 5, 0);
            my_object0.Add(floor3);
            pens.Add(pen0);

            col1 = new mesh();
            col1.vertices.Add(1 * 0.5F, 0, 0, 1); //0
            col1.edges.Add(new int[] { 1, 11 });
            for (int i = 1; i <= 10; i++)
            {
                col1.vertices.Add((float)Math.Cos(i * 0.5236)*0.5F, (float)Math.Sin(i * 0.5236) * 0.5F, 0, 1); //1..10
                col1.edges.Add(new int[] { i + 1, i + 12 });
            }
            col1.vertices.Add((float)Math.Cos(11 * 0.5236) * 0.5F, (float)Math.Sin(11 * 0.5236) * 0.5F, 0, 1); //11
            col1.edges.Add(new int[] { 23 });
            col1.vertices.Add((float)Math.Cos(11 * 0.5236) * 0.5F, (float)Math.Sin(11 * 0.5236) * 0.5F, 10, 1); //12
            col1.edges.Add(new int[] { 13, 23 });
            for (int i = 1; i <= 10; i++)
            {
                col1.vertices.Add((float)Math.Cos(i * 0.5236) * 0.5F, (float)Math.Sin(i * 0.5236) * 0.5F, 10, 1); //13..22
                col1.edges.Add(new int[] { i + 13 });
            }
            col1.vertices.Add((float)Math.Cos(11 * 0.5236) * 0.5F, (float)Math.Sin(11 * 0.5236) * 0.5F, 10, 1); //23
            col1.edges.Add(null);
            col2 = new mesh(col1);
            col1.vertices *= moving_on(15, (float)14.5, 0);
            col2.vertices *= moving_on(15, (float)3.5, 0);
            my_object0.Add(col1);
            my_object0.Add(col2);
            pens.Add(pen0);
            pens.Add(pen0);

            axe_x = new mesh();
            axe_x.vertices.Add(0, 0, 0, 1); //0
            axe_x.edges.Add(new int[] { 1 });
            axe_x.vertices.Add(50, 0, 0, 1); //1
            my_object0.Add(axe_x);
            pens.Add(penx);

            axe_y = new mesh();
            axe_y.vertices.Add(0, 0, 0, 1); //0
            axe_y.edges.Add(new int[] { 1 });
            axe_y.vertices.Add(0, 50, 0, 1); //1
            my_object0.Add(axe_y);
            pens.Add(peny);

            axe_z = new mesh();
            axe_z.vertices.Add(0, 0, 0, 1); //0
            axe_z.edges.Add(new int[] { 1 });
            axe_z.vertices.Add(0, 0, 50, 1); //1
            my_object0.Add(axe_z);
            pens.Add(penz);

            // Название осей


            // X
            axe_x = new mesh();
            axe_x.vertices.Add(22, 0, 1, 1); //0
            axe_x.edges.Add(new int[] { 1 });
            axe_x.vertices.Add(23, 0, 2, 1); //1
            my_object0.Add(axe_x);
            pens.Add(penx2);

            axe_x = new mesh();
            axe_x.vertices.Add(23, 0, 1, 1); //0
            axe_x.edges.Add(new int[] { 1 });
            axe_x.vertices.Add(22, 0, 2, 1); //1
            my_object0.Add(axe_x);
            pens.Add(penx2);

            // Y
            axe_y = new mesh();
            axe_y.vertices.Add(0, 22, 1.5f, 1); //0
            axe_y.edges.Add(new int[] { 1 });
            axe_y.vertices.Add(0, 22.5f, 2, 1); //1
            my_object0.Add(axe_y);
            pens.Add(peny2);

            axe_y = new mesh();
            axe_y.vertices.Add(0, 22, 1.5f, 1); //0
            axe_y.edges.Add(new int[] { 1 });
            axe_y.vertices.Add(0, 21.5f, 2, 1); //1
            my_object0.Add(axe_y);
            pens.Add(peny2);

            axe_y = new mesh();
            axe_y.vertices.Add(0, 22, 1, 1); //0
            axe_y.edges.Add(new int[] { 1 });
            axe_y.vertices.Add(0, 22, 1.5f, 1); //1
            my_object0.Add(axe_y);
            pens.Add(peny2);

            // Z
            axe_z = new mesh();
            axe_z.vertices.Add(0, 1, 22, 1); //0
            axe_z.edges.Add(new int[] { 1 });
            axe_z.vertices.Add(0, 2, 22, 1); //1
            my_object0.Add(axe_z);
            pens.Add(penz2);

            axe_z = new mesh();
            axe_z.vertices.Add(0, 2, 22, 1); //0
            axe_z.edges.Add(new int[] { 1 });
            axe_z.vertices.Add(0, 1, 23, 1); //1
            my_object0.Add(axe_z);
            pens.Add(penz2);

            axe_z = new mesh();
            axe_z.vertices.Add(0, 1, 23, 1); //0
            axe_z.edges.Add(new int[] { 1 });
            axe_z.vertices.Add(0, 2, 23, 1); //1
            my_object0.Add(axe_z);
            pens.Add(penz2);



            mult_all(zooming(13, 13, 13), ref my_object0);
            default_object();
        }
        private void default_object()
        {
            if (my_object != null)
                my_object.Clear();
            my_object = new List<mesh>();
            for (int i = 0; i < my_object0.Count; i++)
                my_object.Add(new mesh(my_object0[i]));
        }
        public void mult_all(matrix A, ref List<mesh> list)
        {
            for (int i = 0; i < list.Count; i++)
                list[i].vertices *= A;
        }
        public void draw_pic()
        {    
            g.Clear(Color.White);
            for (int i = 0; i < my_object.Count; i++)
                my_object[i].draw(ref g, pens[i]);
            pictureBox1.Image = bmp;
        }
        private void button1_Click(object sender, EventArgs e)
        {
            default_object();
            mult_all(Ay, ref my_object);
            draw_pic();
        }
        private void button2_Click(object sender, EventArgs e)
        {
            default_object();
            mult_all(Ax, ref my_object);
            draw_pic();
        }
        private void button3_Click(object sender, EventArgs e)
        {
            default_object();
            mult_all(Az, ref my_object);
            draw_pic();
        }



        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
            is_down = false;
        }
        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            is_down = true;
            cursor_x = e.X;
            cursor_y = e.Y;
        }
        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            int tmp = 30;
            if (is_down)
            {
                int x = e.X - cursor_x;
                int y = e.Y - cursor_y;
                if (x > tmp)
                {
                    mult_all(Rz_minus, ref my_object);
                    cursor_x = e.X;
                    cursor_y += y / 2;
                }
                if (x < -tmp)
                {
                    mult_all(Rz_plus, ref my_object);
                    cursor_x = e.X;
                    cursor_y += y / 2;
                }
                if (y > tmp)
                {
                    if (ModifierKeys.HasFlag(Keys.Shift))
                    {
                        mult_all(Ry_plus, ref my_object);
                    }
                    else mult_all(Rx_plus, ref my_object);
                    cursor_x += x / 2;
                    cursor_y = e.Y;
                }
                if (y < -tmp)
                {
                    if (ModifierKeys.HasFlag(Keys.Shift))
                    {
                        mult_all(Ry_minus, ref my_object);
                    }
                    else mult_all(Rx_minus, ref my_object);
                    cursor_x += x / 2;
                    cursor_y = e.Y;
                }
                draw_pic();
            }
        }
    }
}
