namespace Siren {
    using System;

    using ManyConsole.CommandLineUtils;

    class AudioProgram {
        static int Main(string[] args) {
            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(AudioProgram)),
                args, Console.Out);
        }
    }
}
