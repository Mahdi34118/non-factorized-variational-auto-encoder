from numbers import Number

import torch 
import torch.nn as nn

class NoneActLayer(nn.Module):
    def __init__(self):
        super(NoneActLayer, self).__init__()
        pass
    def forward(self, x):
        return x
    
class XTanhActLayer(nn.Module):
    def __init__(self, coef_x=1., coef_tanh=1.):
        super(XTanhActLayer, self).__init__()
        self.coef_x = coef_x
        self.coef_tanh = coef_tanh
    def forward(self, x):
        return self.coef_tanh * x.tanh() + self.coef_x * x
    
    def __str__(self):
        return f"xtanh(coef_x={self.coef_x}, coef_tanh={self.coef_tanh})"
    def __repr__(self):
        return self.__str__()

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activations):
        super(MLP, self).__init__()
        
        # Dimensions
        assert isinstance(hidden_dims, list), f'Oops!! Wrong argument type for "hidden_dims": {type(hidden_dims)}. "hidden_dims" must be a list.'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Set activation functions 
        if isinstance(activations, str):
            self.activations = [activations] * len(self.hidden_dims)
        elif isinstance(activations, list):
            assert len(activations)==len(hidden_dims)+1, f'Oops!! Wrong argument value for "activations". "activations" must have only one element more than "hidden_dims".'
            self.activations = activations
        else:
            raise ValueError(f'Oops!! Wrong Argument type for "activations": {activations}. "activations" must be one of this types: [str, list]')
        self.activations = MLP.get_activation_functions(self.activations)
        
        # Set the layers
        self.mlp = MLP.get_layers(
                        _input_dim = self.input_dim,
                        _hidden_dims = self.hidden_dims,
                        _output_dim = self.output_dim,
                        _acts = self.activations
                    )
        
    @staticmethod
    def get_activation_functions(acts):
        _activations = list()
        for act in acts:
            act = act.split("_")
            if act[0].lower() == "sigmoid":
                _activations.append(nn.Sigmoid())
            elif act[0].lower() == "xtanh":
                coef_x = 1 if len(act)==1 else float(act[1])
                coef_tanh = 1 if len(act)<3 else float(act[2])
                _activations.append(XTanhActLayer(coef_x=coef_x, coef_tanh=coef_tanh))
            elif act[0].lower() == "relu":
                _activations.append(nn.ReLU())
            elif act[0].lower() == "lrelu":
                _activations.append(nn.LeakyReLU())
            elif act[0].lower() == "none":
                _activations.append(NoneActLayer())
            else:
                raise ValueError(f'Oops!! Wrong argument value for "activations": {acts} -> {act}. "activations" must be one of this values: ["none", sigmoid", "xtanh", "relu", "lrelu"]')

        return _activations

    @staticmethod
    def get_layers(_input_dim, _hidden_dims, _output_dim, _acts):
        layers = list()
        
        layers_in = [_input_dim] + _hidden_dims
        layers_out = _hidden_dims + [_output_dim]

        for (_in, _out, _act) in zip(layers_in, layers_out, _acts):
            layers.append(
                nn.Linear(_in, _out)
            )
            layers.append(_act)
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class MLPDoubleHead(nn.Module):
    def __init__(self, input_dim, head1_dim, head2_dim, hidden_dims, hidden_activations, head1_activation, head2_activation) -> None:
        super().__init__()

        # Dimensions
        assert isinstance(hidden_dims, list), f'Oops!! Wrong argument type for "hidden_dims": {type(hidden_dims)}. "hidden_dims" must be a list.'
        self.input_dim = input_dim
        self.head1_dim = head1_dim
        self.head2_dim = head2_dim
        self.hidden_dims = hidden_dims
        
        self.base_mlp = None
        print(f"hidden_dims: {hidden_dims} {len(hidden_dims)}")
        if len(hidden_dims)!=0:
            self.base_mlp = MLP(
                input_dim = input_dim,
                hidden_dims = hidden_dims[:-1],
                output_dim = hidden_dims[-1],
                activations = hidden_activations 
            )

        head1_activation = MLP.get_activation_functions([head1_activation])[0]
        head2_activation = MLP.get_activation_functions([head2_activation])[0]

        heads_in_dim = self.hidden_dims[-1] if len(self.hidden_dims)!=0 else self.input_dim
        self.head1 = nn.Sequential(
            nn.Linear(heads_in_dim, self.head1_dim),
            head1_activation
        )

        self.head2 = nn.Sequential(
            nn.Linear(heads_in_dim, head2_dim),
            head2_activation
        )

    def forward(self, x):
        rep = x 
        if self.base_mlp != None:
            rep = self.base_mlp(x)

        out_head1 = self.head1(rep)
        out_head2 = self.head2(rep) 

        return out_head1, out_head2


if __name__ == "__main__":
    # a simple mlp
    hidden_1 = [4, 15, 10, 8]
    activations = ["sigmoid", "xtanh_0.1_0.01", "relu", "lrelu", "none"] 
    mlp = MLP(
        input_dim = 10,
        output_dim = 5,
        hidden_dims = hidden_1,
        activations = activations
    )

    print(hidden_1)
    print(mlp)

    print("="*32)
    # a simple mlp with no hidden layers
    mlp_2 = MLP(
        input_dim = 10,
        output_dim = 5,
        hidden_dims = [],
        activations = ["none"]
    )

    print(mlp_2)

    print("="*32)
    # A double head mlp
    hidden_2 = [4, 15, 10, 8]
    activations_2 = ["sigmoid", "xtanh_0.1_0.01", "relu", "lrelu"] 
    
    mlp_dobule_head = MLPDoubleHead(
        input_dim = 10,
        hidden_dims = hidden_2,
        hidden_activations = activations_2,
        head1_dim = 8,
        head1_activation = "sigmoid",
        head2_dim = 5,
        head2_activation = "none"
    )

    print(hidden_2)
    print(mlp_dobule_head)

    print("="*32)
    # A dobule head mlp without any hidden layers
    mlp_dobule_head_2 = MLPDoubleHead(
        input_dim = 10,
        hidden_dims = [],
        hidden_activations = [],
        head1_dim = 8,
        head1_activation = "sigmoid",
        head2_dim = 5,
        head2_activation = "none"
    )

    print(mlp_dobule_head_2)



