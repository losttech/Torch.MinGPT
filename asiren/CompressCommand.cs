namespace LostTech.Torch.NN {
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using ManyConsole.CommandLineUtils;

    using TorchSharp;

    using static TorchSharp.torch.nn;
    using static TorchSharp.torch.optim;

    public class CompressCommand : ConsoleCommand {
        public override int Run(string[] remainingArguments) {
            string sourcePath = remainingArguments[0];
            string destSamplePath = Path.ChangeExtension(sourcePath, ".packed.wav");
            string destPath = Path.ChangeExtension(sourcePath, ".as");
            float[] samples = AudioTools.Read(sourcePath, out int sampleRate);

            // this allows SIREN to oversaturate channels without adding to the loss
            double innerInitWeightLimit = Siren.InnerInitWeightLimit(this.Width);
            Module siren = Sequential(
                //new GaussianNoise(stddev: 1f/(128*1024*64)),
                ("siren", new Siren(inputSize: 1,
                          innerSizes: Enumerable.Repeat(this.Width, this.Layers).ToArray(),
                          inputFrequencyScale: Siren.RecommendedFrequencyScales.SoundInput*20)),
                ("out_dense", Linear(inputSize: this.Width, outputSize: 1)),
                ("clip", ClampToValidChannelValueRange())
                //new GaussianNoise(stddev: 1f/4096),
            );

            var device = torch.cuda.is_available() ? torch.CUDA : null;

            var coords = Enumerable.Range(0, samples.Length)
                .Select(i => i * 2f / samples.Length - 1)
                .ToArray().ToTensor(new long[] { samples.Length, 1 });

            var feedableSamples = samples.ToTensor(new long[] { samples.Length, 1 });

            if (device is not null) {
                siren = siren.to(device);
                coords = coords.to(device);
                feedableSamples = feedableSamples.to(device);
            }

            var optimizer = Adam(siren.parameters(), learningRate: 3e-6);
            var loss = functional.mse_loss();

            if (device is not null) {
                coords = coords.to(device);
                feedableSamples = feedableSamples.to(device);
            }

            int lastUpgrade = 0;
            const int ImproveEvery = 50;
            var improved = ImprovedCallback.Create((sender, eventArgs) => {
                if (eventArgs.Epoch < lastUpgrade + ImproveEvery) return;

                using var noGrad = torch.no_grad();
                siren.save(destPath);

                using var sample = siren.forward(coords).cpu();
                float[] rawSample = new float[samples.Length];
                sample.Data<float>().CopyTo(rawSample);
                for (int i = 0; i < rawSample.Length; i++) {
                    rawSample[i] = Clamp(min: -1, max: 1, sample[i, 0].ToScalar().ToSingle());
                }
                AudioTools.Write(destSamplePath, rawSample, sampleRate);

                Console.WriteLine();
                Console.WriteLine("saved!");

                lastUpgrade = eventArgs.Epoch;
            });

            int batchesPerEpoch = samples.Length / this.BatchSize;

            for (int epoch = 0; epoch < this.Epochs; epoch++) {
                double totalLoss = 0;
                for (int batchN = 0; batchN < batchesPerEpoch; batchN++) {
                    var (ins, outs) = (coords, feedableSamples).RandomBatch(this.BatchSize, device);
                    optimizer.zero_grad();
                    using var predicted = siren.forward(ins);
                    using var batchLoss = loss(predicted, outs);
                    batchLoss.backward();
                    optimizer.step();

                    ins.Dispose(); outs.Dispose();

                    using var noGrad = torch.no_grad();
                    totalLoss += batchLoss.detach().cpu().mean().ToDouble();

                    GC.Collect();
                    Console.Title = $"epoch: {epoch} batch: {batchN} of {batchesPerEpoch}";
                }
                var epochEnd = new EpochEndEventArgs {
                    Epoch = epoch,
                    AvgLoss = totalLoss / batchesPerEpoch,
                };
                improved(null, epochEnd);
            }

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

        static Module ClampToValidChannelValueRange()
            => new Clamp { Min = -1, Max = +1 };
    }
}
