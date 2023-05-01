using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LostTech.Torch.NN;
public static class TextTransformer {
    public static Tensor Sample(Module<Tensor, Tensor> model,
                                int blockSize,
                                Tensor x, int steps,
                                bool alwaysBest = true,
                                float temperature = 1,
                                int? topK = null) {
        //take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        //the sequence, feeding the predictions back into the model each time. Clearly the sampling
        //has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        //of block_size, unlike an RNN that has an infinite context window.
        model.eval();
        for (int k = 0; k < steps; k++) {
            var x_cond = x.size(1) <= blockSize ? x : x[.., ^blockSize..]; // crop context if needed
            var logits = model.forward(x_cond);
            // pluck the logits at the final step and scale by temperature
            logits = logits[.., -1, ..] / temperature;
            // optionally crop probabilities to only the top k options
            if (topK is not null)
                logits = TopKLogits(logits, topK.Value);
            // apply softmax to convert to probabilities
            var probs = functional.softmax(logits, dim: -1);
            // sample from the distribution or take the most likely
            Tensor ix = !alwaysBest
                ? torch.multinomial(probs, num_samples: 1)
                : torch.topk(probs, k: 1, dim: -1).indices;
            // append to the sequence and continue
            x = torch.cat(new[] { x, ix }, dim: 1);
        }

        return x;
    }

    public static byte[] Sample(Module<Tensor, Tensor> model,
                                int blockSize,
                                Dictionary<byte, int> byte2token,
                                Dictionary<int, byte> token2byte,
                                Device? device,
                                IEnumerable<byte> prefix
                                ) {
        using var scope = torch.NewDisposeScope();
        var @in = tensor(prefix.Select(b => byte2token[b]).ToArray(),
                         ScalarType.Int64,
                         device)
                  .unsqueeze(0);
        int[] @out = Sample(model, blockSize, @in, steps: 2000)[0].to(ScalarType.Int32).cpu().data<int>().ToArray();
        return @out.Select(i => token2byte[i]).ToArray();
    }

    static Tensor TopKLogits(Tensor logits, int k) {
        var v = torch.topk(logits, k).values;
        var output = logits.clone();
        output[output < v[.., TensorIndex.Slice(-1)]] = float.NegativeInfinity;
        return output;
    }
}
