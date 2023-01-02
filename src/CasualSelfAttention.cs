namespace LostTech.Torch.NN;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// A vanilla multi-head masked self-attention layer with a projection at the end.
/// It is possible to use
/// <see cref="torch.nn.MultiheadAttention">MultiheadAttention from Torch</see>
/// here but I am including an explicit implementation here to show that there is
/// nothing too scary here.
/// </summary>
public sealed class CasualSelfAttention : Module<Tensor, Tensor> {
    readonly Linear key, query, value;
    readonly Dropout attentionDropout, residualDropout;
    readonly Linear outputProjection;
    readonly int headCount;
    readonly Tensor mask;

    public CasualSelfAttention(int embeddingSize, int headCount, int blockSize,
                               float attentionDropout = 0.1f,
                               float residualDropout = 0.1f): base("CasualSelfAttention") {
        if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));
        if (headCount <= 0) throw new ArgumentOutOfRangeException(nameof(headCount));
        if (embeddingSize % headCount != 0)
            throw new ArgumentException($"{nameof(embeddingSize)} must be evenly divisible by {nameof(headCount)}");

        // TODO: register parts

        this.key = Linear(embeddingSize, embeddingSize);
        this.query = Linear(embeddingSize, embeddingSize);
        this.value = Linear(embeddingSize, embeddingSize);

        this.attentionDropout = Dropout(attentionDropout);
        this.residualDropout = Dropout(residualDropout);

        this.outputProjection = Linear(embeddingSize, embeddingSize);

        this.RegisterComponents();

        this.mask = ones(blockSize, blockSize).tril()
                    .view(1, 1, blockSize, blockSize);
        this.register_buffer("mask", this.mask);

        this.headCount = headCount;
    }

    public override Tensor forward(Tensor x) {
        using var disposeScope = NewDisposeScope();

        var (batchSize, tokens, channels) = x.size();

        var key = this.key.forward(x).view(batchSize, tokens, this.headCount, channels / this.headCount).transpose(1, 2);
        var query = this.query.forward(x).view(batchSize, tokens, this.headCount, channels / this.headCount).transpose(1, 2);
        var value = this.value.forward(x).view(batchSize, tokens, this.headCount, channels / this.headCount).transpose(1, 2);

        // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        var attention = query.matmul(key.transpose(-2, -1)) * (1.0 / Math.Sqrt(key.size(-1)));
        attention = attention.masked_fill(this.mask[.., .., ..(int)tokens, ..(int)tokens] == 0, float.NegativeInfinity);
        attention = functional.softmax(attention, -1);
        attention = this.attentionDropout.forward(attention);

        var @out = attention.matmul(value); // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        // re-assemble all head outputs side by side
        @out = @out.transpose(1, 2).contiguous().view(batchSize, tokens, channels);

        @out = this.residualDropout.forward(this.outputProjection.forward(@out));
        return disposeScope.MoveToOuter(@out);
    }
}
