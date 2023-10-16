from .nets import cleanVAE, cleanIVAE, iVAE, VAE, Discriminator, DiscreteVAE, DiscreteIVAE, permute_dims
from .my_mlp import MLP as MLP
from .my_mlp import MLPDoubleHead as MLPDoubleHead
from .nets import MLP as MLP_ivae
from .wrappers import IVAE_wrapper, TCL_wrapper
from .civae import CIVAE
from .icarl import ICARL