using OpenTK;
using OpenTK.Graphics.OpenGL;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace CrystalGrowthSimulator.Models
{
    public class CrystalFace
    {
        public Vector3[] Vertices { get; set; }
        public Vector3 Normal { get; set; }
        public Color Color { get; set; }
        public float Opacity { get; set; } = 1.0f;

        public CrystalFace(Vector3[] vertices, Vector3 normal, Color color)
        {
            Vertices = vertices;
            Normal = normal;
            Color = color;
        }

        public void Draw()
        {
            GL.Color4(Color.R / 255f, Color.G / 255f, Color.B / 255f, Opacity);
            GL.Begin(PrimitiveType.Polygon);
            GL.Normal3(Normal);

            foreach (var vertex in Vertices)
            {
                GL.Vertex3(vertex);
            }

            GL.End();
        }
    }

    public class RealCrystal
    {
        public Vector3 Position { get; set; }
        public Vector3 TipPosition { get; private set; }
        public float CurrentHeight { get; set; }
        public float GrowthProgress { get; set; } = 0f;
        public List<CrystalFace> Faces { get; private set; }
        public float Rotation { get; set; }
        public bool IsGrowing { get; set; } = true;
        public bool IsDissolving { get; set; } = false;
        public float TargetHeight { get; private set; }
        public float GrowthSpeed { get; private set; }
        public float BaseRadius { get; private set; }
        public Vector3 GrowthDirection { get; private set; }

        private float dissolveProgress = 0f;
        private Random random;

        public RealCrystal(Vector3 position, Random random, List<RealCrystal> existingCrystals)
        {
            Position = position;
            this.random = random;
            Faces = new List<CrystalFace>();

            TargetHeight = 1.2f + (float)random.NextDouble() * 0.8f;
            GrowthSpeed = 0.01f + (float)random.NextDouble() * 0.02f;
            CurrentHeight = 0.1f;
            BaseRadius = 0.06f + (float)random.NextDouble() * 0.04f;

            GrowthDirection = GetRandomGrowthDirection();

            CreateBaseGeometry();
            UpdateTipPosition();
        }

        private Vector3 GetRandomGrowthDirection()
        {
            if (random.NextDouble() < 0.4f)
            {
                return new Vector3(0, 1, 0);
            }
            else
            {
                return new Vector3(
                    (float)(random.NextDouble() - 0.5f) * 1.5f,
                    (float)random.NextDouble() * 0.8f + 0.2f,
                    (float)(random.NextDouble() - 0.5f) * 1.5f
                ).Normalized();
            }
        }

        public void Update(float deltaTime)
        {
            Rotation += deltaTime * (0.3f + (float)random.NextDouble() * 0.7f);

            if (IsGrowing)
            {
                GrowCrystal(deltaTime);
            }
            else if (IsDissolving)
            {
                DissolveCrystal(deltaTime);
            }
        }

        private void GrowCrystal(float deltaTime)
        {
            GrowthProgress += deltaTime * GrowthSpeed;
            GrowthProgress = Math.Min(GrowthProgress, 1.0f);

            CurrentHeight = 0.1f + GrowthProgress * (TargetHeight - 0.1f);

            Faces.Clear();
            CreateBaseGeometry();
            UpdateTipPosition();

            if (GrowthProgress >= 0.99f)
            {
                StartDissolution();
            }
        }

        private void UpdateTipPosition()
        {
            TipPosition = Position + GrowthDirection * CurrentHeight;
        }

        private void DissolveCrystal(float deltaTime)
        {
            dissolveProgress += deltaTime * 0.015f;
            dissolveProgress = Math.Min(dissolveProgress, 1.0f);

            foreach (var face in Faces)
            {
                face.Opacity = 1.0f - dissolveProgress;
            }

            if (dissolveProgress >= 1.0f)
            {
                ResetCrystal();
            }
        }

        private void ResetCrystal()
        {
            IsDissolving = false;
            IsGrowing = true;
            GrowthProgress = 0f;
            dissolveProgress = 0f;
            CurrentHeight = 0.1f;

            TargetHeight = 1.2f + (float)random.NextDouble() * 0.8f;
            GrowthSpeed = 0.01f + (float)random.NextDouble() * 0.02f;
            BaseRadius = 0.06f + (float)random.NextDouble() * 0.04f;
            GrowthDirection = GetRandomGrowthDirection();

            Faces.Clear();
            CreateBaseGeometry();
            UpdateTipPosition();
        }

        private void CreateBaseGeometry()
        {
            CreatePencilCrystal(CurrentHeight, BaseRadius);
        }

        private void CreatePencilCrystal(float height, float baseRadius)
        {
            Faces.Clear();

            int sides = 6;

            Vector3 right, forward;
            if (Math.Abs(GrowthDirection.Y) > 0.9f)
            {
                right = Vector3.UnitX;
                forward = Vector3.UnitZ;
            }
            else
            {
                right = Vector3.Cross(GrowthDirection, Vector3.UnitY).Normalized();
                forward = Vector3.Cross(right, GrowthDirection).Normalized();
            }

            // Основание кристалла
            Vector3[] baseVertices = new Vector3[sides];
            for (int i = 0; i < sides; i++)
            {
                float angle = i * (float)Math.PI * 2 / sides;
                baseVertices[i] =
                    right * ((float)Math.Cos(angle) * baseRadius) +
                    forward * ((float)Math.Sin(angle) * baseRadius);
            }

            // Верхнее основание
            float topRadius = baseRadius * 0.6f;
            Vector3 topCenter = GrowthDirection * height * 0.8f;
            Vector3[] topVertices = new Vector3[sides];
            for (int i = 0; i < sides; i++)
            {
                float angle = i * (float)Math.PI * 2 / sides;
                topVertices[i] = topCenter +
                    right * ((float)Math.Cos(angle) * topRadius) +
                    forward * ((float)Math.Sin(angle) * topRadius);
            }

            // Острый кончик
            Vector3 tipPoint = GrowthDirection * height;

            // Боковые грани
            for (int i = 0; i < sides; i++)
            {
                int nextIndex = (i + 1) % sides;

                Vector3 v1 = baseVertices[i];
                Vector3 v2 = baseVertices[nextIndex];
                Vector3 v3 = topVertices[nextIndex];
                Vector3 v4 = topVertices[i];

                Vector3 normal = CalculateNormal(v1, v2, v3);

                // Средняя высота для градиента
                float avgHeight = (Vector3.Dot(v1, GrowthDirection) +
                                 Vector3.Dot(v2, GrowthDirection) +
                                 Vector3.Dot(v3, GrowthDirection) +
                                 Vector3.Dot(v4, GrowthDirection)) / 4f;
                float heightFactor = avgHeight / height;
                heightFactor = Math.Max(0f, Math.Min(1f, heightFactor));

                Color faceColor = GetGradientColor(heightFactor);

                Faces.Add(new CrystalFace(new[] { v1, v2, v3, v4 }, normal, faceColor));
            }

            // Грани к острию
            for (int i = 0; i < sides; i++)
            {
                int nextIndex = (i + 1) % sides;

                Vector3 v1 = topVertices[i];
                Vector3 v2 = topVertices[nextIndex];
                Vector3 v3 = tipPoint;

                Vector3 normal = CalculateNormal(v1, v2, v3);

                // Высота для верхних граней
                float avgHeight = (Vector3.Dot(v1, GrowthDirection) +
                                 Vector3.Dot(v2, GrowthDirection) +
                                 Vector3.Dot(v3, GrowthDirection)) / 3f;
                float heightFactor = avgHeight / height;
                heightFactor = Math.Max(0f, Math.Min(1f, heightFactor));

                Color faceColor = GetGradientColor(heightFactor);

                Faces.Add(new CrystalFace(new[] { v1, v2, v3 }, normal, faceColor));
            }

            // Нижнее основание
            float baseHeightFactor = 0f; // Самое низкое положение
            Color baseColor = GetGradientColor(baseHeightFactor);
            Faces.Add(new CrystalFace(baseVertices, -GrowthDirection.Normalized(), baseColor));
        }

        private Color GetGradientColor(float heightFactor)
        {
            // Градиент от синего (0.0) к зеленому (1.0)
            heightFactor = Math.Max(0f, Math.Min(1f, heightFactor));

            int r = 0;
            int g = (int)(50 + 205 * heightFactor);   // Зеленый: 50 -> 255
            int b = (int)(255 - 200 * heightFactor);  // Синий: 255 -> 55

            return Color.FromArgb(r, g, b);
        }

        private Vector3 CalculateNormal(Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 normal = Vector3.Cross(b - a, c - b);
            normal.Normalize();
            return normal;
        }

        public void Draw()
        {
            GL.PushMatrix();
            GL.Translate(Position);

            if (GrowthDirection.Length > 0.001f)
            {
                GL.Rotate(Rotation, GrowthDirection.X, GrowthDirection.Y, GrowthDirection.Z);
            }

            foreach (var face in Faces)
            {
                face.Draw();
            }

            GL.PopMatrix();
        }

        public void StartDissolution()
        {
            IsGrowing = false;
            IsDissolving = true;
            dissolveProgress = 0f;
        }

        public void StartGrowth()
        {
            IsDissolving = false;
            IsGrowing = true;
        }

        public bool IsVisible()
        {
            return dissolveProgress < 0.99f;
        }
    }
}