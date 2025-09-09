using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lab2_projection
{
    public class mesh
    {
        public matrix vertices;
        public List<int[]> edges;
        public List<int[]> pics;
        public mesh()
        {
            vertices = new matrix();
            edges = new List<int[]>();
            pics = new List<int[]>();
        }
        public mesh(mesh copy)
        {
            vertices = new matrix(copy.vertices);
            this.edges = copy.edges;
        }
        public void draw(ref Graphics g, Pen pen)
        {
            for (int i = 0; i < edges.Count; i++)
            {
                if (edges[i] != null)
                {
                    for (int j = 0; j < edges[i].Length; j++)
                    {
                        float x1 = vertices.numbers[i][0];
                        float z1 = vertices.numbers[i][2];
                        float x2 = vertices.numbers[edges[i][j]][0];
                        float z2 = vertices.numbers[edges[i][j]][2];
                        g.DrawLine(pen, x1, z1, x2, z2);
                    }
                }
            }
        }
    }
}
