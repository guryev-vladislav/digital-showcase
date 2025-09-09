using CrystalGrowthSimulator.Models;
using CrystalGrowthSimulator.Renderers;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using System;
using System.Drawing;
using System.Windows.Forms;

namespace CrystalGrowthSimulator
{
    public partial class MainForm : Form
    {
        private OpenTK.GLControl glControl;
        private Timer timer;
        private DateTime lastUpdateTime;
        private CrystalCluster crystalCluster;
        private SceneRenderer sceneRenderer;

        private bool isMouseDown = false;
        private Point lastMousePos;
        private float cameraDistance = 10f;
        private float cameraRotationX = 45f;
        private float cameraRotationY = 0f;

        public MainForm()
        {
            InitializeComponent();
            InitializeGLControl();
            InitializeCrystalCluster();
            InitializeSceneRenderer();
            SetupTimer();
            SetupMouseEvents();
            SetupKeyEvents();

            this.Text = "Crystal Cluster - Growing";
        }

        private void InitializeGLControl()
        {
            glControl = new OpenTK.GLControl();
            glControl.Dock = DockStyle.Fill;
            glControl.Paint += GLControl_Paint;
            glControl.Resize += GLControl_Resize;
            Controls.Add(glControl);
        }

        private void InitializeCrystalCluster()
        {
            crystalCluster = new CrystalCluster(new Vector3(0, 1, 0), 7);
            lastUpdateTime = DateTime.Now;
        }

        private void InitializeSceneRenderer()
        {
            sceneRenderer = new SceneRenderer();
        }

        private void SetupTimer()
        {
            timer = new Timer();
            timer.Interval = 16;
            timer.Tick += (s, e) =>
            {
                float deltaTime = (float)(DateTime.Now - lastUpdateTime).TotalSeconds;
                lastUpdateTime = DateTime.Now;

                crystalCluster.Update(deltaTime);
                glControl.Invalidate();
            };
            timer.Start();
        }

        private void SetupMouseEvents()
        {
            glControl.MouseDown += (s, e) =>
            {
                if (e.Button == MouseButtons.Left)
                {
                    isMouseDown = true;
                    lastMousePos = e.Location;
                }
            };

            glControl.MouseUp += (s, e) =>
            {
                if (e.Button == MouseButtons.Left)
                {
                    isMouseDown = false;
                }
            };

            glControl.MouseMove += (s, e) =>
            {
                if (isMouseDown)
                {
                    float deltaX = e.X - lastMousePos.X;
                    float deltaY = e.Y - lastMousePos.Y;

                    cameraRotationY += deltaX * 0.5f;
                    cameraRotationX = Math.Max(10f, Math.Min(80f, cameraRotationX + deltaY * 0.5f));

                    lastMousePos = e.Location;
                    glControl.Invalidate();
                }
            };

            glControl.MouseWheel += (s, e) =>
            {
                cameraDistance += e.Delta * 0.001f;
                cameraDistance = Math.Max(4f, Math.Min(20f, cameraDistance));
                glControl.Invalidate();
            };
        }

        private void SetupKeyEvents()
        {
            this.KeyPreview = true;
            this.KeyDown += MainForm_KeyDown;
        }

        private void GLControl_Resize(object sender, EventArgs e)
        {
            if (glControl == null) return;
            glControl.MakeCurrent();
            GL.Viewport(0, 0, glControl.Width, glControl.Height);
        }

        private void GLControl_Paint(object sender, PaintEventArgs e)
        {
            if (glControl == null) return;

            glControl.MakeCurrent();

            GL.ClearColor(0.1f, 0.1f, 0.15f, 1.0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            SetupProjection();
            SetupCamera();

            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Lighting);
            GL.Enable(EnableCap.Light0);
            GL.Enable(EnableCap.ColorMaterial);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            SetupLighting();

            crystalCluster.Draw();

            DrawPlatform();

            glControl.SwapBuffers();
        }

        private void SetupProjection()
        {
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();

            float aspect = glControl.Width / (float)glControl.Height;
            Matrix4 perspective = Matrix4.CreatePerspectiveFieldOfView(
                MathHelper.PiOver4, aspect, 0.1f, 100f);
            GL.LoadMatrix(ref perspective);
        }

        private void SetupCamera()
        {
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();

            Vector3 cameraPosition = new Vector3(
                (float)(cameraDistance * Math.Sin(MathHelper.DegreesToRadians(cameraRotationX)) *
                        Math.Sin(MathHelper.DegreesToRadians(cameraRotationY))),
                (float)(cameraDistance * Math.Cos(MathHelper.DegreesToRadians(cameraRotationX))),
                (float)(cameraDistance * Math.Sin(MathHelper.DegreesToRadians(cameraRotationX)) *
                        Math.Cos(MathHelper.DegreesToRadians(cameraRotationY)))
            );

            Matrix4 lookAt = Matrix4.LookAt(
                cameraPosition,
                Vector3.Zero, // Смотрим в центр
                Vector3.UnitY
            );
            GL.LoadMatrix(ref lookAt);
        }

        private void SetupLighting()
        {
            GL.Light(LightName.Light0, LightParameter.Diffuse, new float[] { 0.9f, 0.9f, 0.9f, 1.0f });
            GL.Light(LightName.Light0, LightParameter.Ambient, new float[] { 0.3f, 0.3f, 0.3f, 1.0f });
            GL.Light(LightName.Light0, LightParameter.Position, new float[] { 3f, 5f, 3f, 0.0f });

            GL.Material(MaterialFace.Front, MaterialParameter.Specular, new float[] { 0.7f, 0.7f, 0.8f, 1.0f });
            GL.Material(MaterialFace.Front, MaterialParameter.Shininess, 100f);
        }

        private void DrawPlatform()
        {
            GL.PushMatrix();
            GL.Translate(0, 1.0f, 0); // Прямо в центре
            GL.Scale(0.8f, 0.05f, 0.8f); // Маленькая платформа

            GL.Color3(Color.FromArgb(120, 100, 80));
            GL.Begin(PrimitiveType.Quads);

            GL.Normal3(0, 1, 0);
            GL.Vertex3(-1, 0, -1);
            GL.Vertex3(1, 0, -1);
            GL.Vertex3(1, 0, 1);
            GL.Vertex3(-1, 0, 1);

            GL.End();
            GL.PopMatrix();
        }

        private void MainForm_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.H:
                    MessageBox.Show("Controls:\nSpace - Start Dissolution\nR - Reset All Crystals\nMouse - Rotate camera\nWheel - Zoom\nH - Help", "Help");
                    break;
                case Keys.Space:
                    crystalCluster.StartDissolutionForAll();
                    this.Text = "Crystal Cluster - Dissolving";
                    break;
                case Keys.R:
                    crystalCluster.StartGrowthForAll();
                    this.Text = "Crystal Cluster - Growing";
                    break;
            }
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            timer?.Stop();
            base.OnFormClosed(e);
        }
    }
}