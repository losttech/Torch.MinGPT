namespace LostTech.Torch.NN {
    using TorchSharp.NN;
    using TorchSharp.Tensor;

    public class GaussianNoise : CustomModule {
        public double StdDev { get; set; } = 1;
        public GaussianNoise():base("GaussianNoise"){}

        public override TorchTensor forward(TorchTensor t) {
            var noise = t.randn_like();
            noise.mul_(this.StdDev);
            return t + noise;
        }
    }
}