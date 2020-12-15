import torch.nn as nn

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class DynamicsFunction(nn.Module):
    def __init__(self, model, params=None):
        super().__init__()
        self.model = model
        if params is None:
            self.params = {}
        else:
            self.params = params
    
    def forward(self, t, x):
        return self.model(x, **self.params)

    def update_params(self, params):
        self.params.update(params)
