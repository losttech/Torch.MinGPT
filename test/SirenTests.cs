namespace tensorflow.keras {
    using System;
    using System.Linq;
    using numpy;
    using tensorflow.keras.layers;
    using tensorflow.keras.losses;
    using tensorflow.keras.optimizers;
    using Xunit;

    public partial class SirenTests {
        [Fact]
        public Model CanLearn() {
            // requires Internet connection
            (dynamic train, dynamic test) = tf.keras.datasets.fashion_mnist.load_data();
            ndarray trainImage = NormalizeChannelValue(train.Item1)[0];
            trainImage = (ndarray)np.expand_dims(trainImage, axis: 2).reshape(new []{ 28*28, 1 });
            var coords = Coord(28, 28).ToNumPyArray().reshape(new []{28*28, 2});

            var model = new Sequential(new object[] {
                new Siren(2, Enumerable.Repeat(128, 3).ToArray()),
                new Dense(units: 1, activation: tf.keras.activations.relu_fn),
            });

            model.compile(
                optimizer: new Adam(),
                loss: new MeanSquaredError());

            model.fit(coords, targetValues: trainImage, epochs: 2, batchSize: 28*28, stepsPerEpoch: 1024);

            double testLoss = model.evaluate(coords, trainImage);
            return model;
        }

        public static float[,,] Coord(int width, int height) {
            var result = new float[width, height, 2];
            for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                result[x, y, 0] = x*2f/width - 1;
                result[x, y, 1] = y*2f/height - 1;
            }

            return result;
        }

        public static dynamic NormalizeChannelValue(dynamic value) => value / 128f - 1f;
    }
}
