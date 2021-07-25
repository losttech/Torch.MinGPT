namespace LostTech.Torch.NN {
    using System;
    using System.Drawing;
    using System.Drawing.Imaging;

    using static TorchSharp.torch;

    class ImageTools {
        public static float[,,] Coord(int width, int height) {
            float[,,] result = new float[width, height, 2];
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++) {
                    result[x, y, 0] = x * 2f / width - 1;
                    result[x, y, 1] = y * 2f / height - 1;
                }

            return result;
        }

        public static dynamic NormalizeChannelValue(dynamic value) => value / 128f - 1f;

        public static unsafe byte[,,] ToBytesHWC(Bitmap bitmap) {
            byte[,,] result = new byte[bitmap.Height, bitmap.Width, 4];
            int rowBytes = bitmap.Width * 4;
            var rect = new Rectangle(default, new Size(bitmap.Width, bitmap.Height));
            var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            try {
                fixed (byte* dest = result) {
                    for (int y = 0; y < bitmap.Height; y++) {
                        var source = data.Scan0 + data.Stride * y;
                        Buffer.MemoryCopy((byte*)source, destination: &dest[rowBytes * y], rowBytes, rowBytes);
                    }
                }
            } finally {
                bitmap.UnlockBits(data);
            }

            return result;
        }

        public static Tensor PrepareImage(byte[,,] image, Device? device = null) {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            int channels = image.GetLength(2);

            byte[] flattened = image.Flatten();
            var unnormalized = ByteTensor.from(flattened, dimensions: new long[] { height * width, channels });
            if (device is not null) unnormalized = unnormalized.to(device);
            var normalized = ImageTools.NormalizeChannelValue(unnormalized.to_type(ScalarType.Float32));
            return normalized;
        }
    }
}
