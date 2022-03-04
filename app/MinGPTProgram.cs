using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using LostTech.Torch.NN;
using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;


const int BLOCK_SIZE = 64;
const int MAX_EPOCHS = 512;

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
Module gpt = new GPT(vocabularySize: vocab.Count,
                        embeddingSize: 128,
                        blockSize: BLOCK_SIZE,
                        headCount: 8,
                        blockCount: 10);

var device = torch.cuda.is_available() ? torch.CUDA : null;

if (device is not null) gpt = gpt.to(device);

int batchSize = torch.cuda.is_available() ? 4 * 1024 : 64;

// lowered learning rate to avoid destabilization
var optimizer = torch.optim.AdamW(gpt.parameters(), learningRate: 0.0003);
var lossF = torch.nn.functional.cross_entropy_loss();

if (args.Length == 0) {
    gpt.load("sample.weights");
    throw new NotImplementedException("sample");
}

long step = 0;
var epochStopwatch = Stopwatch.StartNew();

for(int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
epochStopwatch.Restart();
double epochLoss = 0;

foreach (string filePath in args) {
    byte[] data = File.ReadAllBytes(filePath);

    for (int position = 0; position < data.Length - BLOCK_SIZE; position++) {
        using var _ = torch.NewDisposeScope();
        byte[] chunk = data[position..(position + BLOCK_SIZE + 1)];
        int[] indexes = chunk.Select(b => btoi[b]).ToArray();
        var @in = tensor(indexes[..^1], dtype: ScalarType.Int64);
        var @out = tensor(indexes[1..], dtype: ScalarType.Int64);

        var logits = gpt.forward(@in);
        var loss = lossF.Invoke(logits.view(-1, logits.size(-1)), @out.view(-1));

        loss = loss.mean();

        utils.clip_grad_norm_(gpt.parameters(), 1);

        optimizer.step();
        optimizer.zero_grad();

        step++;

        using var noGrad = no_grad();
        epochLoss += loss.detach().cpu().mean().ToDouble();

        Console.Title = $"epoch: {epoch} batch: {position + 1} of {data.Length - BLOCK_SIZE}";
    }
}
}
