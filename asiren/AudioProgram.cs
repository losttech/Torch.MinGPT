namespace Siren {
    using System;
    using LostTech.Gradient;
    using LostTech.TensorFlow;
    using ManyConsole.CommandLineUtils;
    class AudioProgram {
        static int Main(string[] args) {
            GradientEngine.UseEnvironmentFromVariable();
            TensorFlowSetup.Instance.EnsureInitialized();

            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(AudioProgram)),
                args, Console.Out);
        }
    }
}
