namespace LostTech.Torch {
    using TorchSharp;
    using TorchSharp.Tensor;
    static class TensorTools {
        public static (TorchTensor ins, TorchTensor outs) RandomBatch(
                this (TorchTensor ins, TorchTensor outs) pair, int batchSize,
                Device? device = null) {
            using var noGrad = new AutoGradMode(false);
            using var indices = Int64Tensor.randint(max: pair.ins.shape[0],
                                                    size: new long[] { batchSize },
                                                    device: device);
            var tensorIndices = TorchTensorIndex.Tensor(indices);
            return (pair.ins[tensorIndices], pair.outs[tensorIndices]);
        }
    }
}