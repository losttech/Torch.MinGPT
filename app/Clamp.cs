namespace LostTech.Torch {
    using TorchSharp.NN;
    using TorchSharp.Tensor;

    public class Clamp : CustomModule {
        public double Min { get; set; }
        public double Max { get; set; }
        public Clamp() : base("clip") { }

        public override TorchTensor forward(TorchTensor t)
            => t.clamp(this.Min, this.Max);
    }
}