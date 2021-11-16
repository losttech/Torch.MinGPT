namespace LostTech.Torch {
    using static TorchSharp.torch;

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
    }
}