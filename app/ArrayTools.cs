namespace LostTech.Torch.NN {
    using System;

    using TorchSharp;
    using static TorchSharp.torch;

    static class ArrayTools {
        public static Tensor ToTensor(this float[,,] array)
            => Flatten(array).ToTensor(new long[]{
                array.GetLongLength(0),
                array.GetLongLength(1),
                array.GetLongLength(2)});

        public static T[] Flatten<T>(this T[,,] array) {
            int d0 = array.GetLength(0);
            int d1 = array.GetLength(1);
            int d2 = array.GetLength(2);
            var result = new T[d0 * d1 * d2];
            for (int i = 0; i < array.GetLength(0); i++)
                for (int j = 0; j < array.GetLength(1); j++)
                    for (int k = 0; k < array.GetLength(2); k++)
                        result[i * d1 * d2 + j * d2 + k] = array[i, j, k];
            return result;
        }

        public static void Deconstruct<T>(this T[] array, out T i0, out T i1, out T i2) {
            i0 = array[0];
            i1 = array[1];
            i2 = array[2];
        }
    }
}