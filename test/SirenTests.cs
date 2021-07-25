namespace LostTech.Torch.NN {
    using System.Diagnostics;
    using System.Drawing;
    using System.Runtime.InteropServices;

    using TorchSharp;

    using Xunit;
    using static ImageTools;
    using static TorchSharp.torch.nn;

    public partial class SirenTests {
        [Fact]
        public Module CanLearn() {
            torch.random.manual_seed(119);

            var thisAssembly = System.Reflection.Assembly.GetExecutingAssembly();
            string wikiLogoName = thisAssembly.GetManifestResourceNames()[0];

            using var wikiLogo = new Bitmap(thisAssembly.GetManifestResourceStream(wikiLogoName));
            byte[,,] bytesHWC = ToBytesHWC(wikiLogo);
            var trainImage = PrepareImage(bytesHWC);
            var coords = Coord(wikiLogo.Height, wikiLogo.Width).Flatten()
                         .ToTensor(new long[] { wikiLogo.Height * wikiLogo.Width, 2 });

            var model = Sequential(
                ("siren", new Siren(2, innerSizes: new[] { 128, 128, 128, 128 })),
                ("linear", Linear(inputSize: 128, outputSize: 1)),
                ("final_activation", ReLU())
            );

            using var optimizer = torch.optim.Adam(model.parameters());
            var loss = functional.mse_loss();

            const int batchSize = 1024;
            const int batches = 1000;

            for (int batchN = 0; batchN < batches; batchN++) {
                var (ins, outs) = (coords, trainImage).RandomBatch(batchSize);
                optimizer.zero_grad();
                using var predicted = model.forward(ins);
                using var batchLoss = loss(predicted, outs);
                batchLoss.backward();
                optimizer.step();

                ins.Dispose();
                outs.Dispose();

                if (batchN % (batches / 10) == (batches / 10 - 1))
                    Trace.WriteLine($"loss {batchN}: {batchLoss.mean().ToSingle()}");
            }
            using var recall = model.forward(coords);
            double recallLoss = loss(recall, trainImage).mean().ToDouble();
            Assert.True(recallLoss < 0.0001, recallLoss.ToString());
            return model;
        }
    }
}
