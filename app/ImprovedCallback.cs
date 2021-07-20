namespace LostTech.Torch {
    using System;
    using System.Collections.Generic;

    static class ImprovedCallback {
        public static EventHandler<EpochEndEventArgs> Create(EventHandler<EpochEndEventArgs> onLossImproved) {
            double bestLoss = double.PositiveInfinity;
            void on_epoch_end(object? _, EpochEndEventArgs args) {
                if (args.AvgLoss < bestLoss) {
                    bestLoss = args.AvgLoss;
                    onLossImproved?.Invoke(null, args);
                }
            }
            return on_epoch_end;
        }
    }

    delegate void Callback(int epoch, double avgLoss);

    class EpochEndEventArgs : EventArgs {
        public int Epoch { get; set; }
        public double AvgLoss { get; set; }
    }
}
