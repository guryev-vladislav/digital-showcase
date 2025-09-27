using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lab2_projection
{
    public class matrix
    {
        public int height;
        public List<List<float>> numbers;
        public matrix()
        {
            height = 0;
            numbers = new List<List<float>>();
        }
        public matrix(int height)
        {
            this.height = height;
            numbers = new List<List<float>>();
            for (int i = 0; i < this.height; i++)
            {
                numbers.Add(new List<float>());
                for (int j = 0; j <= 3; j++)
                {
                    numbers[i].Add(new float());
                }
            }
        }
        public matrix(matrix copy) : this(copy.height)
        {
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j <= 3; j++)
                {
                    numbers[i][j] = copy.numbers[i][j];
                }
            }
        }
        public void Add(float x, float y, float z, float p)
        {
            numbers.Add(new List<float>());
            height++;
            numbers[height - 1].Add(x);
            numbers[height - 1].Add(y);
            numbers[height - 1].Add(z);
            numbers[height - 1].Add(p);
        }
        public static matrix operator *(matrix A, matrix B)
        {
            matrix res = new matrix(A.height);
            if (4 == B.height)
            {
                for (int i = 0; i < res.height; i++)
                {
                    for (int j = 0; j <= 3; j++)
                    {
                        res.numbers[i][j] = (float)0;
                        for (int k = 0; k <= 3; k++)
                        {
                            res.numbers[i][j] += A.numbers[i][k] * B.numbers[k][j];
                        }
                    }
                }
            }
            return res;
        }
        public static matrix operator +(matrix A, matrix B)
        {
            matrix res = new matrix(A.height);
            if (A.height == B.height)
            {
                for (int i = 0; i < res.height; i++)
                {
                    for (int j = 0; j <= 3; j++)
                    {
                        res.numbers[i][j] = A.numbers[i][j] + B.numbers[i][j];
                    }
                }
            }
            return res;
        }
        public static matrix operator *(matrix A, float num)
        {
            matrix res = new matrix(A.height);
            for (int i = 0; i < res.height; i++)
            {
                for (int j = 0; j <= 3; j++)
                {
                    res.numbers[i][j] = A.numbers[i][j] * num;
                }
            }
            return res;
        }
    }
}
