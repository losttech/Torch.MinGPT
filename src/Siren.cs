namespace LostTech.Torch.NN {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using TorchSharp;
    using TorchSharp.Modules;

    using static System.FormattableString;
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    /// <summary>
    /// Implements <a href="https://vsitzmann.github.io/siren/">SIREN: Implicit Neural Representations with Periodic Activation Functions</a>
    /// - a network with sine activations.
    ///
    /// <para>All SIREN inputs must be normalized to [-1;1] interval.</para>
    /// </summary>
    public class Siren : CustomModule {
        readonly Linear[] innerLayers;
        /// <summary>
        /// Frequency scale for input layer (<c>first_omega_0</c> in the paper).
        /// <para>The layer formula is <c>sin(FreqScale*(Weight*input+Bias)</c></para>
        /// </summary>
        public float InnerFrequencyScale { get; }
        /// <summary>
        /// Frequency scale for intermediate layers (<c>omega_0</c> in the paper).
        /// <para>The intermediate layer formula is <c>sin(FreqScale*(Weight*prevOutput+Bias)</c></para>
        /// </summary>
        public float InputFrequencyScale { get; }

        /// <summary>Frequency scales, recommended in the paper ("omega_0")</summary>
        public static class RecommendedFrequencyScales {
            internal const float Default = 30;

            /// <summary>Frequency scale for inner layers, any data type</summary>
            public static readonly float Inner = 30;
            /// <summary>Frequency scale for input layer for images</summary>
            public static readonly float ImageInput = 30;
            /// <summary>Frequency scale for input layer for sound (raw wav)</summary>
            public static readonly float SoundInput = 3000;
        }

        /// <summary>
        /// Creates a new <see cref="Siren"/> of the specified size.
        /// </summary>
        /// <param name="inputSize">
        /// Number of inputs.
        /// This MUST be set to the correct value, otherwise the network might not be trainable.
        /// </param>
        /// <param name="innerSizes">
        /// Sizes of inner layers.
        /// Must be a non-empty array of positive numbers.</param>
        /// <param name="inputFrequencyScale">
        /// Frequency scale for input layer (<c>first_omega_0</c> in the paper).
        /// <para>The layer formula is <c>sin(FreqScale*(Weight*input+Bias)</c></para>
        /// <para>See <see cref="RecommendedFrequencyScales"/> for good values.</para>
        /// </param>
        /// <param name="innerFrequencyScale">
        /// Frequency scale for intermediate layers (<c>omega_0</c> in the paper).
        /// <para>The intermediate layer formula is <c>sin(FreqScale*(Weight*prevOutput+Bias)</c></para>
        /// </param>
        public Siren(int inputSize, int[] innerSizes,
                     float inputFrequencyScale = RecommendedFrequencyScales.Default,
                     float innerFrequencyScale = RecommendedFrequencyScales.Default)
                     : base("Siren") {
            if (inputSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(inputSize));
            if (innerSizes is null || innerSizes.Length == 0)
                throw new ArgumentNullException(nameof(innerSizes));
            if (innerSizes.Any(size => size < 0))
                throw new ArgumentOutOfRangeException(nameof(innerSizes));
            if (!IsValidFrequencyScale(inputFrequencyScale))
                throw new ArgumentOutOfRangeException(nameof(inputFrequencyScale));
            if (!IsValidFrequencyScale(innerFrequencyScale))
                throw new ArgumentOutOfRangeException(nameof(innerFrequencyScale));

            this.InputFrequencyScale = inputFrequencyScale;
            this.InnerFrequencyScale = innerFrequencyScale;

            using var noGrad = torch.no_grad();

            this.innerLayers = new Linear[innerSizes.Length];

            int currentInputSize = inputSize;
            for (int innerIndex = 0; innerIndex < innerSizes.Length; innerIndex++) {
                // This is the crucial part of the paper.
                // Without proper weight initialization training will fail.
                // See paper sec. 3.2
                double weightLimits = innerIndex > 0
                    ? InnerInitWeightLimit(currentInputSize, frequencyScale: this.InnerFrequencyScale)
                    : 1.0f / inputSize;
                var layer = Linear(inputSize: currentInputSize, outputSize: innerSizes[innerIndex]);
                this.innerLayers[innerIndex] = layer;
                init.uniform(layer.Weight, low: -weightLimits, high: +weightLimits);
                this.RegisterModule(Invariant($"i{innerIndex}"), layer);

                currentInputSize = innerSizes[innerIndex];
            }
        }

        public override Tensor forward(Tensor t) {
            var result = t;
            for (int layerIndex = 0; layerIndex < this.innerLayers.Length; layerIndex++) {
                var layer = this.innerLayers[layerIndex];
                float frequencyScale = layerIndex == 0
                    ? this.InputFrequencyScale
                    : this.InnerFrequencyScale;
                var next = layer.forward(result);
                next.mul_(frequencyScale);
                next.sin_();
                if (!ReferenceEquals(result, t))
                    result.Dispose();
                result = next;
            }

            return result;
        }

        static bool IsValidFrequencyScale(float scale)
            => !float.IsInfinity(scale)
            && !float.IsNaN(scale)
            && (Math.Abs(scale) > 4 * float.Epsilon);

        public static double InnerInitWeightLimit(int inputSize, float frequencyScale = RecommendedFrequencyScales.Default)
            => Math.Sqrt(6.0f / inputSize) / frequencyScale;
    }
}
