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
            // Кристаллы на платформе с небольшим разбросом
            // Платформа находится на Y = 1.0f, поэтому ставим кристаллы прямо на нее
            Vector3 position = centerPosition + new Vector3(
                (float)(random.NextDouble() - 0.5) * 0.12f,
                1.0f, // Прямо на платформе! (Y = 1.0f)
                (float)(random.NextDouble() - 0.5) * 0.12f
            );

            var crystal = new RealCrystal(position, random, Crystals);
            crystal.GrowthProgress = (float)random.NextDouble() * 0.2f;
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