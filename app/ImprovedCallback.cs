namespace tensorflow.keras {
    using System;
    using System.Collections.Generic;
    using LostTech.Gradient;
    using tensorflow.keras.callbacks;

    static class ImprovedCallback {
        public static Callback Create(EventHandler<EpochEndEventArgs> onLossImproved) {
            double bestLoss = double.PositiveInfinity;
            void on_epoch_end(int epoch, IDictionary<string, dynamic> logs) {
                if (logs["loss"] < bestLoss) {
                    bestLoss = logs["loss"];
                    onLossImproved?.Invoke(null, new EpochEndEventArgs {
                        Epoch = epoch,
                        Logs = logs,
                    });
                }
            }
            var callbackFn = new Action<int, IDictionary<string, dynamic>>(on_epoch_end);
            // using LabmdaCallback is more performant, as it does not require TF to call on_batch_begin
            return new LambdaCallback(on_epoch_end: PythonFunctionContainer.Of(callbackFn));
        }
    }

    class EpochEndEventArgs : EventArgs {
        public int Epoch { get; set; }
        public IDictionary<string, dynamic> Logs { get; set; }
    }
}
