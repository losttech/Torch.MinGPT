namespace LostTech.Torch.NN;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

static class ModuleTools {
    public static void Register<T>(this Module parent, out T var, T module,
                                   [CallerArgumentExpression("var")] string name = null!)
        where T : Module {
        if (name is null) throw new ArgumentNullException(nameof(name));

        var = module;
        parent.register_module(name, var);
    }
}
