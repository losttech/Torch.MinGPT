namespace LostTech.Torch {
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    public class Clamp : Module<Tensor, Tensor> {
        public double Min { get; set; }
        public double Max { get; set; }
        public Clamp() : base("clip") { }

        public override Tensor forward(Tensor t)
            => t.clamp(this.Min, this.Max);
    }
}