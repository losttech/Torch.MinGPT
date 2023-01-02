namespace LostTech.Torch.NN {
    using System.Diagnostics;
    using System.Drawing;
    using System.Linq;
    using System.Runtime.CompilerServices;

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

            const int IntermediateSize = 128;

            var model = Sequential(
                //("siren", new GPT(2, innerSizes: Enumerable.Repeat(IntermediateSize, 3).ToArray())),
                ("linear", Linear(inputSize: IntermediateSize, outputSize: 4)),
                ("final_activation", GELU())
            );

            using var optimizer = torch.optim.Adam(model.parameters());
            var loss = MSELoss();

            int batchSize = wikiLogo.Height * wikiLogo.Width;
            const int batches = 40;

            for (int batchN = 0; batchN < batches; batchN++) {
                var (ins, outs) = (coords, trainImage).RandomBatch(batchSize);
                optimizer.zero_grad();
                using var predicted = model.forward(ins);
                using var batchLoss = loss.forward(predicted, outs);
                batchLoss.backward();
                optimizer.step();

                ins.Dispose();
                outs.Dispose();

                if (batchN % (batches / 10) == (batches / 10 - 1))
                    Trace.WriteLine($"loss {batchN}: {batchLoss.mean().ToSingle()}");
            }
            using var recall = model.forward(coords);
            double recallLoss = loss.forward(recall, trainImage).mean().ToDouble();
            Assert.True(recallLoss < 0.25, recallLoss.ToString());
            return model;
        }

        [ModuleInitializer]
        internal static void Setup() {
            // workaround for https://github.com/dotnet/TorchSharp/issues/449
            //torch.TryInitializeDeviceType(DeviceType.CPU);
        }
    }
}
