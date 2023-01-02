namespace LostTech.Torch {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    static class TensorTools {
        public static (Tensor ins, Tensor outs) RandomBatch(
                this (Tensor ins, Tensor outs) pair, int batchSize,
                Device? device = null) {
            using var noGrad = no_grad();
            using var indices = randint(high: pair.ins.shape[0],
                                        size: new long[] { batchSize },
                                        dtype: ScalarType.Int64,
                                        device: device);
            var tensorIndices = TensorIndex.Tensor(indices);
            return (pair.ins[tensorIndices], pair.outs[tensorIndices]);
        }

        public static Tensor BatchForward(this Module<Tensor, Tensor> module, Tensor ins, int batchSize) {
            var inChunks = ins.split(batchSize);

            var outChunks = inChunks.Select(c => {
                var tmpOut = module.forward(c);
                c.Dispose();
                tmpOut.detach_().Dispose();
                return tmpOut;
            }).ToList();

            var output = cat(outChunks, dim: 0);
            foreach (var chunk in outChunks)
                chunk.Dispose();

            return output;
        }
    }
}