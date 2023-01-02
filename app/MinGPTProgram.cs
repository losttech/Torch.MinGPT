using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

using LostTech.Torch.NN;

using ShellProgressBar;

using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;


const int BLOCK_SIZE = 64;
const int MAX_EPOCHS = 1;

torch.random.manual_seed(42);

var vocab = new HashSet<byte> { 0 };
if (args.Length == 0) args = new[] { Environment.GetEnvironmentVariable("ET_DATASET") };
args.AsParallel().ForAll(input => {
    var fileSet = new HashSet<byte>(File.ReadAllBytes(input));
    lock (vocab) {
        vocab.UnionWith(fileSet);
    }
});

byte[] itob = vocab.ToArray();
var btoi = Enumerable.Range(0, vocab.Count).ToDictionary(i => itob[i], i => i);
var gpt = new GPT(vocabularySize: vocab.Count,
                     embeddingSize: 128,
                     blockSize: BLOCK_SIZE,
                     headCount: 8,
                     blockCount: 10);

var device = torch.cuda.is_available() ? torch.CUDA : null;

if (device is not null) gpt = gpt.to(device);

int batchSize = torch.cuda.is_available() ? 512 : 64;

// lowered learning rate to avoid destabilization
var optimizer = torch.optim.AdamW(gpt.parameters(), lr: 0.0003);
var lossF = CrossEntropyLoss();

if (args.Length == 0) {
    gpt.load("sample.weights");
    throw new NotImplementedException("sample");
}

long step = 0;
var epochStopwatch = Stopwatch.StartNew();

var displayOptions = new ProgressBarOptions {
    ShowEstimatedDuration = true,
    EnableTaskBarProgress = true,
};
using var progressBar = new ProgressBar(MAX_EPOCHS, "training", displayOptions);
for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
    epochStopwatch.Restart();

    Console.Write($"epoch {epoch}: ");

    double epochLoss = 0;
    int epochBatches = 0;
    foreach (string filePath in args) {
        epochLoss += TrainOnFile(filePath, progressBar, out int batches);
        epochBatches += batches;
    }

    progressBar.Tick($"loss: {epochLoss / epochBatches:0.00}");
}
progressBar.Dispose();

var itobDict = new Dictionary<int, byte>();
for (int i = 0; i < itob.Length; i++) {
    itobDict[i] = itob[i];
}

byte[] prefix = Encoding.Latin1.GetBytes("Hello, ");
byte[] suffix = TextTransformer.Sample(gpt, BLOCK_SIZE,
                                       btoi, itobDict,
                                       device, prefix);
Console.WriteLine(Encoding.Latin1.GetString(suffix));

double TrainOnFile(string filePath, ProgressBar parentProgressBar, out int batches) {
    byte[] data = File.ReadAllBytes(filePath);
    var tokens = tensor(data.Select(b => btoi[b]).ToArray(), ScalarType.Int64, device: device);

    (Tensor, Tensor) GetBatch(int index) {
        var starts = arange(index, index + batchSize,
                            ScalarType.Int64,
                            device);
        starts = unsqueeze(starts, dim: 1);
        var indices = arange(0, BLOCK_SIZE, ScalarType.Int64, device);
        indices = torch.unsqueeze(indices, dim: 0);
        indices = starts + indices;
        var x = tokens[indices];
        var y = tokens[indices + 1];
        return (x, y);
    }

    double totalLoss = 0;
    batches = data.Length / batchSize;

    var displayOptions = new ProgressBarOptions {
        ShowEstimatedDuration = true,
    };
    using var progressBar = parentProgressBar.Spawn(batches, "", displayOptions);
    for (int batchIndex = 0; batchIndex < batches; batchIndex++) {
        using var _ = torch.NewDisposeScope();
        var (@in, @out) = GetBatch(batchIndex);

        var logits = gpt.forward(@in);
        var loss = lossF.forward(logits.view(-1, logits.size(-1)), @out.view(-1));

        loss = loss.mean();
        loss.backward();

        nn.utils.clip_grad_norm_(gpt.parameters(), 1);

        optimizer.step();
        optimizer.zero_grad();

        step++;

        using var noGrad = no_grad();
        totalLoss += loss.detach().cpu().mean().ToDouble();

        progressBar.Tick($"loss: {totalLoss / (batchIndex + 1):0.00}");
    }

    return totalLoss;
}
