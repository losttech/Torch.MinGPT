namespace LostTech.Torch.NN {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;

    using TorchSharp;

    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    class UpscaleProgram {
        static void Main(string[] args) {
            using var sirenModule = new Siren(2, Enumerable.Repeat(64, 4).ToArray());
            using var clipModule = ClampToValidChannelValueRange();
            Module siren = Sequential(
                ("in_noise", new GaussianNoise { StdDev = 1f / (128 * 1024) }),
                ("siren", sirenModule),
                ("out_dense", Linear(inputSize: 64, outputSize: 4)),
                // this allows SIREN to oversaturate channels without adding to the loss
                ("clip", clipModule),
                ("out_noise", new GaussianNoise { StdDev = 1f / 128 })
            );

            var device = torch.cuda.is_available() ? torch.CUDA : null;

            if (device is not null) siren = siren.to(device);

            int batchSize = torch.cuda.is_available() ? 4*1024 : 64;

            // lowered learning rate to avoid destabilization
            var optimizer = torch.optim.Adam(siren.parameters(), learningRate: 0.00032 * 64 / batchSize);
            var loss = torch.nn.functional.mse_loss();

            if (args.Length == 0) {
                siren.load("sample.weights");
                Render(siren, 1034 * 3, 1536 * 3, "sample6X.png");
                return;
            }

            foreach (string imagePath in args) {
                using var original = new Bitmap(imagePath);
                byte[,,] image = ImageTools.ToBytesHWC(original);
                int height = image.GetLength(0);
                int width = image.GetLength(1);
                int channels = image.GetLength(2);
                Debug.Assert(channels == 4);

                using var imageSamples = ImageTools.PrepareImage(image, device);

                var coords = ImageTools.Coord(height, width)
                    .Flatten().ToTensor(new long[] { width * height, 2 });

                var upscaleCoords = ImageTools.Coord(height * 2, width * 2).ToTensor();

                if (device is not null) {
                    coords = coords.to(device);
                    upscaleCoords = upscaleCoords.to(device);
                }

                int lastUpgrade = 0;
                const int ImproveEvery = 500;
                var improved = ImprovedCallback.Create((sender, eventArgs) => {
                    if (eventArgs.Epoch < lastUpgrade + ImproveEvery) return;

                    var upscaled = siren.forward(
                        upscaleCoords.reshape(height * width * 4, 2));
                    upscaled = upscaled.reshape(height * 2, width * 2, channels);
                    using var bitmap = ToImage(RestoreImage(upscaled.cpu()));
                    bitmap.Save("sample4X.png", ImageFormat.Png);

                    siren.save("sample.weights");

                    Console.WriteLine();
                    Console.WriteLine("saved!");

                    lastUpgrade = eventArgs.Epoch;
                });

                int batchCount = height * width / batchSize;
                var epochStopwatch = Stopwatch.StartNew();
                for (int epoch = 0; epoch < 10000; epoch++) {
                    double totalLoss = 0;
                    epochStopwatch.Restart();
                    for (int batchN = 0; batchN < batchCount; batchN++) {
                        var (ins, outs) = (coords, imageSamples).RandomBatch(batchSize, device);
                        optimizer.zero_grad();
                        using var predicted = siren.forward(ins);
                        using var batchLoss = loss(predicted, outs);
                        batchLoss.backward();
                        optimizer.step();

                        using var noGrad = no_grad();
                        using var cpuBatchLoss = batchLoss.cpu();
                        using var meanLoss = cpuBatchLoss.mean();
                        totalLoss += meanLoss.ToDouble();

                        ins.Dispose();
                        outs.Dispose();

                        Console.Title = $"epoch: {epoch} batch: {batchN+1} of {batchCount}";
                    }

                    GC.Collect();
                    Console.WriteLine($"Epoch {epoch}. Avg. loss: {totalLoss / batchCount} in {epochStopwatch.ElapsedMilliseconds}ms");
                    var epochEnd = new EpochEndEventArgs { Epoch = epoch, AvgLoss = totalLoss / batchCount };
                    improved(null, epochEnd);
                }
            }
        }

        static void Render(Module siren, int width, int height, string path) {
            var renderCoords = ImageTools.Coord(height, width).ToTensor();
            var renderBytes = siren.forward(renderCoords.reshape(height * width, 2));
            const int channels = 4;
            renderBytes = renderBytes.reshape(height, width, channels);
            using var bitmap = ToImage(RestoreImage(renderBytes));
            bitmap.Save(path, ImageFormat.Png);
        }

        static Module ClampToValidChannelValueRange()
            => new Clamp {
                Min = ImageTools.NormalizeChannelValue(-0.01f),
                Max = ImageTools.NormalizeChannelValue(255.01f),
            };

        static unsafe byte[,,] RestoreImage(Tensor learnedImage) {
            (long height, long width, long channels) = learnedImage.shape;
            var bytes = (learnedImage * 128f + 128f).clip(0, 255).to_type(ScalarType.Byte).Data<byte>();
            Debug.Assert(bytes.Length == height * width * channels);
            byte[,,] result = new byte[height, width, channels];
            fixed (byte* dest = result)
            fixed (byte* source = bytes)
                Buffer.MemoryCopy(source: source, destination: dest, bytes.Length, bytes.Length);
            return result;
        }

        static unsafe Bitmap ToImage(byte[,,] bytesHWC) {
            if (bytesHWC.GetLength(2) != 4)
                throw new NotSupportedException();
            var bitmap = new Bitmap(bytesHWC.GetLength(1), bytesHWC.GetLength(0));
            int rowBytes = bitmap.Width * 4;
            var rect = new Rectangle(default, new Size(bitmap.Width, bitmap.Height));
            var data = bitmap.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            try {
                fixed (byte* source = bytesHWC) {
                    for (int y = 0; y < bitmap.Height; y++) {
                        var dest = data.Scan0 + data.Stride * y;
                        Buffer.MemoryCopy(&source[rowBytes * y], destination: (byte*)dest, rowBytes, rowBytes);
                    }
                }
            } finally {
                bitmap.UnlockBits(data);
            }

            return bitmap;
        }
    }
}
