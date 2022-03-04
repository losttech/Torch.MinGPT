namespace LostTech.Torch.NN;

using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class TransformerBlock : Module {
    readonly LayerNorm layerNorm1, layerNorm2;
    readonly Module attention;
    readonly Sequential mlp;
    public TransformerBlock(int embeddingSize, Module attention, float residualDropout = 0.1f)
        : base(nameof(TransformerBlock)) {
        if (attention is null) throw new ArgumentNullException(nameof(attention));
        if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));

        // TODO: register parts

        this.layerNorm1 = LayerNorm(new long[] { embeddingSize });
        this.layerNorm2 = LayerNorm(new long[] { embeddingSize });
        this.attention = attention;
        this.mlp = Sequential(
            Linear(embeddingSize, 4 * embeddingSize),
            GELU(),
            Linear(4 * embeddingSize, embeddingSize),
            Dropout(residualDropout)
        );
    }

    public override Tensor forward(Tensor x) {
        using var scope = torch.NewDisposeScope();
        x += attention.forward(layerNorm1.forward(x));
        x += mlp.forward(layerNorm2.forward(x));
        return scope.MoveToOuter(x);
    }
}
