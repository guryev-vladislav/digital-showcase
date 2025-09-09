using OpenTK;
using System;
using System.Collections.Generic;

namespace CrystalGrowthSimulator.Models
{
    public class CrystalCluster
    {
        public List<RealCrystal> Crystals { get; private set; }
        private Random random;
        private Vector3 centerPosition;

        public CrystalCluster(Vector3 center, int crystalCount = 7)
        {
            centerPosition = center;
            random = new Random();
            Crystals = new List<RealCrystal>();

            InitializeCrystals(crystalCount);
        }

        private void InitializeCrystals(int count)
        {
            Crystals.Clear();

            for (int i = 0; i < count; i++)
            {
                AddNewCrystal();
            }
        }

        public void Update(float deltaTime)
        {
            foreach (var crystal in Crystals)
            {
                crystal.Update(deltaTime);
            }

            Crystals.RemoveAll(c => !c.IsVisible());

            while (Crystals.Count < 7)
            {
                AddNewCrystal();
            }
        }

        private void AddNewCrystal()
        {
            // Кристаллы прямо на платформе с небольшим разбросом
            Vector3 position = centerPosition + new Vector3(
                (float)(random.NextDouble() - 0.5) * 0.08f, // Очень маленький разброс
                0f, // Прямо на платформе
                (float)(random.NextDouble() - 0.5) * 0.08f
            );

            var crystal = new RealCrystal(position, random, Crystals);
            crystal.GrowthProgress = (float)random.NextDouble() * 0.3f;
            Crystals.Add(crystal);
        }

        public void Draw()
        {
            foreach (var crystal in Crystals)
            {
                crystal.Draw();
            }
        }

        public void StartDissolutionForAll()
        {
            foreach (var crystal in Crystals)
            {
                crystal.StartDissolution();
            }
        }

        public void StartGrowthForAll()
        {
            foreach (var crystal in Crystals)
            {
                crystal.StartGrowth();
            }
        }
    }
}