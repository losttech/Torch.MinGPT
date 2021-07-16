namespace Siren {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using NAudio.Wave;

    class AudioTools {
        public static float[] Read(string filePath, out int sampleRate) {
            ISampleProvider sampler = new AudioFileReader(filePath);
            if (sampler.WaveFormat.Channels != 1) {
                Console.Error.WriteLine($"warning: downsampling from {sampler.WaveFormat.Channels} channels to mono");
                sampler = sampler.ToMono();
            }

            sampleRate = sampler.WaveFormat.SampleRate;

            var result = new List<float>();
            float[] buffer = new float[sampler.WaveFormat.AverageBytesPerSecond];
            for (int read = sampler.Read(buffer, 0, buffer.Length);
                read > 0;
                read = sampler.Read(buffer, 0, buffer.Length)) {
                result.AddRange(buffer.Take(read));
            }
            return result.ToArray();
        }

        public static void Write(string filePath, float[] samples, int sampleRate) {
            var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
            using var writer = new WaveFileWriter(filePath, format);
            writer.WriteSamples(samples, 0, samples.Length);
        }
    }
}
