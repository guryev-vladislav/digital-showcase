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
            GL.Color4(Color.R, Color.G, Color.B, Opacity);
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

        private float dissolveProgress = 0f;
        private Random random;

        public RealCrystal(Vector3 position, Random random, List<RealCrystal> existingCrystals)
        {
            Position = position;
            this.random = random;
            Faces = new List<CrystalFace>();

            // Параметры для прямых колонн
            TargetHeight = 1.5f + (float)random.NextDouble() * 1.0f; // 1.5 - 2.5
            GrowthSpeed = 0.01f + (float)random.NextDouble() * 0.02f;
            CurrentHeight = 0.1f;
            BaseRadius = 0.08f + (float)random.NextDouble() * 0.04f; // 0.08 - 0.12

            CreateBaseGeometry();
            UpdateTipPosition();
        }

        public void Update(float deltaTime)
        {
            Rotation += deltaTime * (0.5f + (float)random.NextDouble() * 1f);

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

            UpdateGeometry();
            UpdateTipPosition();

            if (GrowthProgress >= 0.99f)
            {
                StartDissolution();
            }
        }

        private void UpdateTipPosition()
        {
            TipPosition = Position + new Vector3(0, CurrentHeight, 0);
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

            TargetHeight = 1.5f + (float)random.NextDouble() * 1.0f;
            GrowthSpeed = 0.01f + (float)random.NextDouble() * 0.02f;
            BaseRadius = 0.08f + (float)random.NextDouble() * 0.04f;

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
            // 8-гранная колонна (как у карандаша)
            int sides = 8;
            Vector3[] baseVertices = new Vector3[sides];
            Vector3[] topVertices = new Vector3[sides];

            // Основание
            for (int i = 0; i < sides; i++)
            {
                float angle = i * (float)Math.PI * 2 / sides;
                baseVertices[i] = new Vector3(
                    (float)Math.Cos(angle) * baseRadius,
                    0,
                    (float)Math.Sin(angle) * baseRadius
                );
            }

            // Верхняя часть (немного уже)
            float topRadius = baseRadius * 0.7f;
            for (int i = 0; i < sides; i++)
            {
                float angle = i * (float)Math.PI * 2 / sides;
                topVertices[i] = new Vector3(
                    (float)Math.Cos(angle) * topRadius,
                    height * 0.8f,
                    (float)Math.Sin(angle) * topRadius
                );
            }

            // Острый кончик
            Vector3 tipPoint = new Vector3(0, height, 0);

            // Боковые грани основания
            for (int i = 0; i < sides; i++)
            {
                int nextIndex = (i + 1) % sides;

                Vector3 v1 = baseVertices[i];
                Vector3 v2 = baseVertices[nextIndex];
                Vector3 v3 = topVertices[nextIndex];
                Vector3 v4 = topVertices[i];

                Vector3 normal = CalculateNormal(v1, v2, v3);
                Faces.Add(new CrystalFace(new[] { v1, v2, v3, v4 }, normal, GetCrystalColor()));
            }

            // Грани к острию
            for (int i = 0; i < sides; i++)
            {
                int nextIndex = (i + 1) % sides;

                Vector3 v1 = topVertices[i];
                Vector3 v2 = topVertices[nextIndex];
                Vector3 v3 = tipPoint;

                Vector3 normal = CalculateNormal(v1, v2, v3);
                Faces.Add(new CrystalFace(new[] { v1, v2, v3 }, normal, GetCrystalColor()));
            }

            // Основание (нижняя грань)
            Faces.Add(new CrystalFace(baseVertices, new Vector3(0, -1, 0), GetCrystalColor()));
        }

        private void UpdateGeometry()
        {
            Faces.Clear();
            CreatePencilCrystal(CurrentHeight, BaseRadius);
        }

        private Color GetCrystalColor()
        {
            int variation = random.Next(-10, 10);
            return Color.FromArgb(
                Math.Max(0, Math.Min(255, 180 + variation)),
                Math.Max(0, Math.Min(255, 200 + variation)),
                Math.Max(0, Math.Min(255, 220 + variation))
            );
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
            GL.Rotate(Rotation, 0, 1, 0);

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