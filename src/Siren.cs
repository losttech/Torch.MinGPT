namespace tensorflow.keras {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using LostTech.Gradient.ManualWrappers;

    using tensorflow.keras.layers;

    public class Siren : Model {
        readonly Dense[] innerLayers;
        public float FrequencyScale { get; }

        public Siren(int inputSize, int[] innerSizes, float frequencyScale = 30.0f) {
            if (inputSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(inputSize));
            if (innerSizes is null || innerSizes.Length == 0)
                throw new ArgumentNullException(nameof(innerSizes));
            if (innerSizes.Any(size => size < 0))
                throw new ArgumentOutOfRangeException(nameof(innerSizes));
            if (float.IsInfinity(frequencyScale) || float.IsNaN(frequencyScale)
                || Math.Abs(frequencyScale) <= 4*float.Epsilon)
                throw new ArgumentOutOfRangeException(nameof(frequencyScale));

            this.FrequencyScale = frequencyScale;

            this.innerLayers = new Dense[innerSizes.Length];

            int currentInputSize = inputSize;
            for (int innerIndex = 0; innerIndex < innerSizes.Length; innerIndex++) {
                double weightLimits = innerIndex > 0
                    ? Math.Sqrt(6.0f / currentInputSize) / this.FrequencyScale
                    : 1.0f / inputSize;
                this.innerLayers[innerIndex] = new Dense(innerSizes[innerIndex],
                    kernel_initializer: new initializers.uniform(minval: -weightLimits, maxval: +weightLimits)
                );
                this.Track(this.innerLayers[innerIndex]);

                currentInputSize = innerSizes[innerIndex];
            }
        }

        Tensor CallImpl(IGraphNodeBase input, object? mask) {
            if (mask != null)
                throw new NotImplementedException("mask");
            var result = (Tensor)input;
            foreach (var innerLayer in this.innerLayers)
                result = tf.sin(innerLayer.__call__(result) * this.FrequencyScale);
            return result;
        }

        public override Tensor call(IGraphNodeBase inputs, IGraphNodeBase training, IGraphNodeBase mask)
            => this.CallImpl(inputs, mask);

        public override Tensor call(IGraphNodeBase inputs, bool training, IGraphNodeBase? mask = null)
            => this.CallImpl(inputs, mask);

        public override Tensor call(IGraphNodeBase inputs, IGraphNodeBase? training = null, IEnumerable<IGraphNodeBase>? mask = null)
            => this.CallImpl(inputs, mask);

        public override TensorShape compute_output_shape(TensorShape input_shape) {
            var outputShape = input_shape.as_list();
            outputShape[^1] = this.innerLayers[^1].units;
            return new TensorShape(outputShape);
        }
    }
}
