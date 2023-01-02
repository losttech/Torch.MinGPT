namespace LostTech.Torch.NN;

using System.Runtime.CompilerServices;

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
