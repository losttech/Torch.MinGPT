namespace Siren {
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using LostTech.Gradient;
    using ManyConsole.CommandLineUtils;
    using numpy;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.callbacks;
    using tensorflow.keras.initializers;
    using tensorflow.keras.layers;
    using tensorflow.keras.optimizers;

    public class CompressCommand : ConsoleCommand {
        public override int Run(string[] remainingArguments) {
            string sourcePath = remainingArguments[0];
            string destSamplePath = Path.ChangeExtension(sourcePath, ".packed.wav");
            string destPath = Path.ChangeExtension(sourcePath, ".as");
            float[] samples = AudioTools.Read(sourcePath, out int sampleRate);

            // this allows SIREN to oversaturate channels without adding to the loss
            var clampToValidChannelRange = PythonFunctionContainer.Of<Tensor, Tensor>(ClampToValidChannelValueRange);

            double innerInitWeightLimit = Siren.InnerInitWeightLimit(this.Width);
            object[] layers = new object[] {
                new InputLayer(input_shape: 1),
                //new GaussianNoise(stddev: 1f/(128*1024*64)),
                new Siren(inputSize: 1,
                          innerSizes: Enumerable.Repeat(this.Width, this.Layers).ToArray(),
                          inputFrequencyScale: Siren.RecommendedFrequencyScales.SoundInput*20),
                new Dense(units: 1, activation: clampToValidChannelRange),
                //new GaussianNoise(stddev: 1f/4096),
            };
            var siren = new Sequential(layers);
            var trainedSiren = new Sequential(layers);

            siren.compile(
                optimizer: new Adam(learning_rate: 3e-6),
                //optimizer: new SGD(learning_rate: 1e-4),
                loss: "mse");

            var coords = Enumerable.Range(0, samples.Length)
                .Select(i => i * 2f / samples.Length - 1)
                .ToNumPyArray().reshape(new[] { samples.Length, 1 });

            var feedableSamples = samples.ToNumPyArray().reshape(new[] { samples.Length, 1 });

            int lastUpgrade = 0;
            const int ImproveEvery = 50;
            var improved = ImprovedCallback.Create((sender, eventArgs) => {
                if (eventArgs.Epoch < lastUpgrade + ImproveEvery) return;

                siren.save_weights(destPath + "+opt");
                trainedSiren.save_weights(destPath);

                ndarray<float> sample = siren.predict(coords, batch_size: this.BatchSize).AsType<float>();
                float[] rawSample = new float[samples.Length];
                for (int i = 0; i < rawSample.Length; i++) {
                    rawSample[i] = Clamp(min: -1, max: 1, sample[i, 0].AsScalar());
                }
                AudioTools.Write(destSamplePath, rawSample, sampleRate);

                Console.WriteLine();
                Console.WriteLine("saved!");

                lastUpgrade = eventArgs.Epoch;
            });


            siren.fit(coords, feedableSamples, epochs: this.Epochs, batchSize: this.BatchSize,
                    shuffleMode: TrainingShuffleMode.Epoch,
                    callbacks: new ICallback[] { improved });

            return 0;
        }

        public double LearningRate => 0.02 / this.BatchSize;
        public int Layers { get; set; } = 4;
        public int Width { get; set; } = 512;
        public int BatchSize { get; set; } = 32 * 1024;
        public int Epochs { get; set; } = 10000;

        public CompressCommand() {
            this.IsCommand("compress");
            this.HasOption("l|layers=", "How many hidden layers must be used",
                (int count) => this.Layers = count);
            this.HasOption("w|width=", "How wide the network should be",
                (int size) => this.Width = size);
            this.HasOption("b|batch=", "Number of points to train on at once",
                (int size) => this.BatchSize = size);
            this.HasOption("e|epochs=", "Number of epochs to try improving quality for",
                (int count) => this.Epochs = count);
            this.SkipsCommandSummaryBeforeRunning();
            this.HasAdditionalArguments(count: 1, "<audio file>");
        }

        static float Clamp(float min, float max, float value)
            => MathF.Max(min, MathF.Min(max, value));

        static Tensor ClampToValidChannelValueRange(Tensor input)
            => tf.clip_by_value(input,
                clip_value_min: -1.000001f,
                clip_value_max: +1.000001f);
    }
}
