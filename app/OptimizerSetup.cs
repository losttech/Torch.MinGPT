namespace LostTech.Torch.NN;

using System;
using System.Collections.Generic;
using System.Linq;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch.nn;

class OptimizerSetup {
    public AdamW Configure(Module transformer) {
        var decaying = new HashSet<string>();
        var nonDecaying = new HashSet<string>();

        var weightWitelist = new HashSet<Type> { typeof(Linear) };
        var weightBlacklist = new HashSet<Type> { typeof(LayerNorm), typeof(Embedding) };

        foreach (var (moduleName, module) in transformer.named_modules())
            foreach (var (paramName, parameter) in module.named_parameters()) {
                string fullName = string.IsNullOrEmpty(moduleName) ? paramName : $"{moduleName}.{paramName}";

                if (paramName.EndsWith("bias"))
                    nonDecaying.Add(fullName);
                else if (paramName.EndsWith("weight") && weightWitelist.Contains(module.GetType()))
                    decaying.Add(fullName);
                else if (paramName.EndsWith("weight") && weightBlacklist.Contains(module.GetType()))
                    nonDecaying.Add(fullName);
            }

        nonDecaying.Add("pos_emb");

        // validate that we considered every parameter
        var allParams = transformer.named_parameters().ToDictionary(kv => kv.name, kv => kv.parameter);
        var intersectionParams = decaying.Intersect(nonDecaying);
        var unionParams = decaying.Union(nonDecaying);
        if (intersectionParams.Any()) throw new NotSupportedException(string.Join(", ", intersectionParams));
        var nonProcessed = allParams.Keys.ToHashSet();
        nonProcessed.ExceptWith(unionParams);
        if (nonProcessed.Count > 0) throw new NotSupportedException(string.Join(", ", nonProcessed));

        var paramGroups = new AdamW.ParamGroup[] {
            new (decaying.Select(n => allParams[n]), weight_decay: 0.1),
            new (nonDecaying.Select(n => allParams[n]), weight_decay: 0),
        };

        return torch.optim.AdamW(paramGroups, lr: 6e-4, beta1: 0.9, beta2: 0.95);
    }
}
