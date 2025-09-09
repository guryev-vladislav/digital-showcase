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
        public Vector3 Orientation { get; private set; }
        public float BaseRadius { get; private set; }

        private float dissolveProgress = 0f;
        private Random random;

        public RealCrystal(Vector3 position, Random random, List<RealCrystal> existingCrystals)
        {
            Position = position;
            this.random = random;
            Faces = new List<CrystalFace>();

            // Меньшие размеры для маленькой платформы
            TargetHeight = 0.3f + (float)random.NextDouble() * 0.4f; // 0.3 - 0.7
            GrowthSpeed = 0.02f + (float)random.NextDouble() * 0.04f;
            CurrentHeight = 0.02f;
            BaseRadius = 0.06f + (float)random.NextDouble() * 0.04f; // 0.06 - 0.1

            Orientation = GetValidOrientation(existingCrystals);

            CreateBaseGeometry();
            UpdateTipPosition();
        }

        private Vector3 GetValidOrientation(List<RealCrystal> existingCrystals)
        {
            int maxAttempts = 10;
            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                Vector3 candidateOrientation = GetRandomOrientation();

                if (!IntersectsWithOtherCrystals(candidateOrientation, existingCrystals))
                {
                    return candidateOrientation;
                }
            }

            return new Vector3(0, 1, 0);
        }

        private bool IntersectsWithOtherCrystals(Vector3 orientation, List<RealCrystal> otherCrystals)
        {
            if (otherCrystals == null || otherCrystals.Count == 0)
                return false;

            Vector3 predictedTip = Position + orientation * TargetHeight;

            foreach (var otherCrystal in otherCrystals)
            {
                if (otherCrystal == this) continue;

                float tipDistance = (predictedTip - otherCrystal.TipPosition).Length;
                float minDistance = (BaseRadius + otherCrystal.BaseRadius) * 2f;

                if (tipDistance < minDistance)
                {
                    return true;
                }
            }

            return false;
        }

        private Vector3 GetRandomOrientation()
        {
            if (random.NextDouble() < 0.6f)
            {
                return new Vector3(0, 1, 0);
            }
            else
            {
                return new Vector3(
                    (float)(random.NextDouble() - 0.5f) * 0.8f,
                    (float)random.NextDouble() * 0.5f + 0.5f,
                    (float)(random.NextDouble() - 0.5f) * 0.8f
                ).Normalized();
            }
        }

        public void Update(float deltaTime)
        {
            Rotation += deltaTime * (1f + (float)random.NextDouble() * 4f);

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

            CurrentHeight = 0.02f + GrowthProgress * (TargetHeight - 0.02f);

            UpdateGeometry();
            UpdateTipPosition();

            if (GrowthProgress >= 0.99f)
            {
                StartDissolution();
            }
        }

        private void UpdateTipPosition()
        {
            TipPosition = Position + Orientation * CurrentHeight;
        }

        private void DissolveCrystal(float deltaTime)
        {
            dissolveProgress += deltaTime * 0.02f;
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
            CurrentHeight = 0.02f;

            TargetHeight = 0.3f + (float)random.NextDouble() * 0.4f;
            GrowthSpeed = 0.02f + (float)random.NextDouble() * 0.04f;
            BaseRadius = 0.06f + (float)random.NextDouble() * 0.04f;

            Faces.Clear();
            CreateBaseGeometry();
            UpdateTipPosition();
        }

        private void CreateBaseGeometry()
        {
            CreatePointedCrystal(CurrentHeight, BaseRadius);
        }

        private void CreatePointedCrystal(float height, float baseRadius)
        {
            Vector3 growthDirection = Orientation * height;
            Vector3[] baseVertices = new Vector3[6];

            // Основание - шестиугольник
            for (int i = 0; i < 6; i++)
            {
                float angle = i * (float)Math.PI * 2 / 6;
                baseVertices[i] = new Vector3(
                    (float)Math.Cos(angle) * baseRadius,
                    0,
                    (float)Math.Sin(angle) * baseRadius
                );
            }

            // Заостренный кончик - вершина пирамиды
            Vector3 tipPoint = growthDirection * 1.2f; // Делаем кончик острым

            // Боковые грани (треугольники к кончику)
            for (int i = 0; i < 6; i++)
            {
                int nextIndex = (i + 1) % 6;

                Vector3 v1 = baseVertices[i];
                Vector3 v2 = baseVertices[nextIndex];
                Vector3 v3 = tipPoint;

                Vector3 normal = CalculateNormal(v1, v2, v3);
                Faces.Add(new CrystalFace(new[] { v1, v2, v3 }, normal, GetCrystalColor()));
            }

            // Основание (шестиугольник)
            Faces.Add(new CrystalFace(baseVertices, -Orientation.Normalized(), GetCrystalColor()));
        }

        private void UpdateGeometry()
        {
            Faces.Clear();
            CreatePointedCrystal(CurrentHeight, BaseRadius);
        }

        private Color GetCrystalColor()
        {
            int variation = random.Next(-15, 15);
            return Color.FromArgb(
                Math.Max(0, Math.Min(255, 200 + variation)),
                Math.Max(0, Math.Min(255, 220 + variation)),
                Math.Max(0, Math.Min(255, 240 + variation))
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
            GL.Rotate(Rotation, Orientation.X, Orientation.Y, Orientation.Z);

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