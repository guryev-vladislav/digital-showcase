using CrystalGrowthSimulator.Models;
using OpenTK.Graphics.OpenGL;
using System.Collections.Generic;

namespace CrystalGrowthSimulator.Renderers
{
    public class SceneRenderer
    {
        private List<RealCrystal> crystals = new List<RealCrystal>();

        public void AddCrystal(RealCrystal crystal)
        {
            crystals.Add(crystal);
        }

        public void ClearCrystals()
        {
            crystals.Clear();
        }

        public void Render()
        {
            foreach (var crystal in crystals)
            {
                crystal.Draw();
            }
        }
    }
}