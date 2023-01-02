namespace LostTech.Torch.NN;

using System;
using System.Collections.Generic;
using System.Linq;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// the full GPT language model, with a context size of <see cref="GPT.BlockSize"/>
/// </summary>
public sealed class GPT : Module<Tensor, Tensor> {
    readonly Embedding tokenEmbedding;
    readonly Parameter positionalEmbedding;
    readonly Dropout dropout;
    readonly Sequential blocks;
    readonly LayerNorm finalNorm;
    readonly Linear decoderHead;
    public int BlockSize { get; }
    public int EmbeddingSize { get; }
    public int BlockCount { get; }
    public int HeadCount { get; }

    public GPT(int vocabularySize, int blockSize, int blockCount, int embeddingSize,
               int headCount,
               float attentionDropout = 0.1f,
               float residualDropout = 0.1f,
               float embeddingDropout = 0.1f)
        : base("MinGPT") {
        if (blockCount <= 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        this.BlockSize = blockSize;
        this.BlockCount = blockCount;
        this.EmbeddingSize = embeddingSize;
        this.HeadCount = headCount;

        this.Register(out this.tokenEmbedding, Embedding(vocabularySize, embeddingSize));
        this.positionalEmbedding = Parameter(zeros(1, this.BlockSize, embeddingSize));
        this.Register(out this.dropout, Dropout(embeddingDropout));

        this.Register(out this.blocks, Sequential(
            Enumerable.Range(0, blockCount)
            .Select(_ => new TransformerBlock(
                embeddingSize: embeddingSize,
                new CasualSelfAttention(
                    embeddingSize: embeddingSize,
                    headCount: headCount,
                    blockSize: blockSize,
                    attentionDropout: attentionDropout,
                    residualDropout: residualDropout)))
        ));

        this.Register(out this.finalNorm, LayerNorm(new long[] { embeddingSize }));
        this.Register(out this.decoderHead, Linear(embeddingSize, vocabularySize, hasBias: false));

        this.apply(InitWeights);
    }

    public override Tensor forward(Tensor index) {
        long tokens = index.size(-1);
        if (tokens > this.BlockSize) throw new ArgumentException("Cannot forward, model block size is exhausted.");

        var tokenEmbeddings = this.tokenEmbedding.forward(index); // each index maps to a (learnable) vector
        var positionEmbeddings = this.positionalEmbedding[.., ..(int)tokens, ..]; // each position maps to a (learnable) vector
        var x = this.dropout.forward(tokenEmbeddings + positionEmbeddings);
        x = this.blocks.forward(x);
        x = this.finalNorm.forward(x);
        var logits = this.decoderHead.forward(x);

        return logits;
    }

    public static Tensor Loss(Tensor outputs, Tensor targets)
        => functional.cross_entropy(
                   outputs.view(-1, outputs.size(-1)),
                   targets.view(-1));

    static void InitWeights(Module module) {
        switch (module) {
        case Linear l:
            init.normal_(l.weight, mean: 0.0, std: 0.02);
            if (l.bias is not null)
                init.zeros_(l.bias);
            break;
        case Embedding e:
            init.normal_(e.weight, mean: 0, std: 0.02);
            break;
        case LayerNorm ln:
            init.zeros_(ln.get_parameter("bias"));
            init.ones_(ln.get_parameter("weight"));
            break;
        case CasualSelfAttention _:
        case TransformerBlock _:
        case Sequential _:
        case GPT _:
            //foreach (var (_, nested) in module.named_children())
            //    nested.apply(InitWeights);
            break;
        case Dropout _:
        case GELU _:
            break;
        default:
            break;
        }
    }


}
