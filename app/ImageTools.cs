namespace tensorflow.keras {
    class ImageTools {
        public static float[,,] Coord(int width, int height) {
            var result = new float[width, height, 2];
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++) {
                    result[x, y, 0] = x * 2f / width - 1;
                    result[x, y, 1] = y * 2f / height - 1;
                }

            return result;
        }

        public static dynamic NormalizeChannelValue(dynamic value) => value / 128f - 1f;
    }
}
