namespace LostTech.Torch.NN {
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    public class GaussianNoise : CustomModule {
        public double StdDev { get; set; } = 1;
        public GaussianNoise():base("GaussianNoise"){}

        public override Tensor forward(Tensor t) {
            var noise = t.randn_like();
            noise.mul_(this.StdDev);
            return t + noise;
        }
    }
}