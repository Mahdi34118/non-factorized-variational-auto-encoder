import numpy as np 

import torch
from torch import nn
import torch.autograd as autograd
from torch.nn import functional as F
from torch import distributions as dist

from .my_mlp import MLP
from .nets import log_normal

class CIVAE(nn.Module):
    # def __init__(self, data_dim, latent_dim, env_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
    def __init__(self,
                 data_encoder,
                 env_encoder,
                 latent_dim,  
                 env_encoder_linear_prior,
                 env_encoder_non_linear_prior,
                 graph_weights_net,
                 mutual_effects_net,
                 hidden_dims=[50],
                 activations='xtanh_0.1',
                 nf_ivae=False
                ):
        super().__init__()
        self.nf_ivae = nf_ivae

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activations = activations

        # encoders
        self.data_encoder = data_encoder
        self.env_encoder = env_encoder
        self.encoder_mu = MLP(
            input_dim = data_encoder.output_dim + env_encoder.output_dim,
            output_dim = latent_dim,
            hidden_dims = hidden_dims,
            activations=activations
        )
        self.encoder_logv = MLP(
            input_dim = data_encoder.output_dim + env_encoder.output_dim,
            output_dim = latent_dim,
            hidden_dims = hidden_dims,
            activations = activations
        )

        # Linear Prior's Environment Encoder
        self.env_encoder_linear_prior = env_encoder_linear_prior

        # Non-Linear Prior's Environment Encoder
        self.env_encoder_non_linear_prior = env_encoder_non_linear_prior
        
        # Causal Graph Weight Calculator
        self.graph_weights = graph_weights_net

        # Causal Graph Mutual Effect Calculator
        self.mutual_effects = mutual_effects_net

        # Decoder
        self.decoder_mu = MLP(
            input_dim = latent_dim, 
            output_dim = self.data_encoder.input_dim, 
            hidden_dims = hidden_dims,
            activations=activations
        )
        self.decoder_var = .1 * torch.ones(1)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        x_rep = self.data_encoder(x)
        u_rep = self.env_encoder(u)
        xu_rep = torch.cat((x_rep, u_rep), 1)

        mu = self.encoder_mu(xu_rep)
        logv = self.encoder_logv(xu_rep)
        return mu, logv.exp()

    def decoder(self, z):
        x = self.decoder_mu(z)
        return x

    def forward(self, x, u):
        mean, var = self.encoder(x, u)
        z = self.reparameterize(mean, var)
        x_hat = self.decoder(z)
        return x_hat, mean, var, z
    
    def linear_prior(self, z, u):
        env_encoding = self.env_encoder_linear_prior(u)
        # env_encoding = torch.exp(env_encoding)

        return (env_encoding * z).sum(dim=-1)
    
    def non_linear_prior(self, z, u):
        env_encoding = self.env_encoder_non_linear_prior(u)
        env_encoding = torch.exp(env_encoding)

        return torch.log(env_encoding * (z**2)).sum(dim=-1)
    
    def weight_effect_corelation(self, z, u):
        
        weights = self.graph_weights(u)

        if self.nf_ivae :
            effects = self.mutual_effects(z)

        else :
        
        # weights = torch.exp(weights)
            pairs = []
            effects = torch.zeros(weights.shape, device=z.device)
            counter = 0
            for ix in range(z.shape[1]):
                other_features = [j for j in range(z.shape[1]) if j!=ix]
            
                for j in other_features:
                    # pairs.append((ix,j))
            # for pair in pairs :
                    f1 = torch.Tensor(z[:,ix:ix+1]).to(z.device)
                    f2 = torch.Tensor(z[:,j:j+1]).to(z.device)

                    index_f1 = torch.Tensor([ix] * f1.shape[0])
                    index_f1 = index_f1[:,None].to(z.device)
                    
                    index_f2 = torch.Tensor([j] * f1.shape[0])
                    index_f2 = index_f2[:,None].to(z.device)

                    zn = torch.cat((f1,f2,index_f1 , index_f2), 1).to(z.device)

                    # import pdb;pdb.set_trace()
                    effects[:,counter:counter+1] = self.mutual_effects(zn)
                    counter += 1

        return (weights * effects).sum(dim=-1)
    
    def weight_matrix_2D(self, u):
        num_rows, num_cols = self.latent_dim, self.latent_dim

        weights_2D = torch.zeros((u.shape[0], num_rows, num_cols), dtype=torch.float64)

        weights_1D = self.graph_weights(u)

        weight_index = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if i==j:
                    continue
                weights_2D[:, i, j] = weights_1D[:, weight_index]
                weight_index += 1

        return weights_2D.to(u.device)
    
    def dag_loss(self, z, u):
        weights = self.weight_matrix_2D(u)
        # print(weights.shape)
        # import pdb;pdb.set_trace()

        # weights = weights / weights.sum(dim=[1,2])[:, None, None]

        # tf.linalg.trace(tf.linalg.expm(self.B * self.B)) - self.d

        # loss = torch.diagonal( 
        #     torch.matrix_exp(weights * weights), 
        #     dim1=-2, 
        #     dim2=-1).sum(-1).to(u.device)

        # loss = z - weights @ z
        # loss = loss.sum(-1)
        loss = torch.norm( z - torch.matmul(weights,z[:,:,None])[:, :, 0] ).to(z.device)

        return loss 

        # effects = torch.cat([effects, z, z**2], dim=1)
        # pz_cu = (weights * effects).sum(dim=-1)
        # return torch.log(pz_cu)
        # return pz_cu

    def log_prob_z_c_u(self, z, u):
        # print(f"linear_prior(z,u): {self.linear_prior(z,u)}")
        # print(f"non_linear_prior(z,u): {self.non_linear_prior(z,u)}")
        # print(f"weight_effect_corelation(z,u): {self.weight_effect_corelation(z,u)}")

        
        logpz_cu = self.linear_prior(z,u) - self.non_linear_prior(z,u) + self.weight_effect_corelation(z,u)
        logpz_cu *= -1

        # print(f"logpz_cu: {logpz_cu}")
        # print("="*32)

        # return torch.log(logpz_cu)
        return logpz_cu

    
    def freeze_network(self, model, freeze=True):
        for param in model.parameters():
            param.requires_grad = freeze

        return model
    
    def optimization_freezer(self, phase):
        if phase==2:
            # UnFreeze the encoder and decoder and freeze other networks

            # Freeze the linear-prior network
            self.env_encoder_linear_prior = self.freeze_network(self.env_encoder_linear_prior, freeze=True)

            # Freeze the non-linear-prion network
            self.env_encoder_non_linear_prior = self.freeze_network(self.env_encoder_non_linear_prior, freeze=True)

            # Freeze the mutual effect network
            self.mutual_effects = self.freeze_network(self.mutual_effects, freeze=True)

            # Freeze the graph weights network
            self.graph_weights = self.freeze_network(self.graph_weights, freeze=True)
        
            # Unfreeze the encoder
            self.encoder_mu = self.freeze_network(self.encoder_mu, freeze=False)
            self.encoder_logv = self.freeze_network(self.encoder_logv, freeze=False)

            # Unfreeze the decoder
            self.decoder_mu = self.freeze_network(self.decoder_mu, freeze=False)
        else: 
            # Freeze the encoder and decoder and unfreeze other networks

            # Freeze the linear-prior network
            self.env_encoder_linear_prior = self.freeze_network(self.env_encoder_linear_prior, freeze=False)

            # Freeze the non-linear-prion network
            self.env_encoder_non_linear_prior = self.freeze_network(self.env_encoder_non_linear_prior, freeze=False)

            # Freeze the mutual effect network
            self.mutual_effects = self.freeze_network(self.mutual_effects, freeze=False)

            # Freeze the graph weights network
            self.graph_weights = self.freeze_network(self.graph_weights, freeze=False)
        
            # Unfreeze the encoder
            self.encoder_mu = self.freeze_network(self.encoder_mu, freeze=True)
            self.encoder_logv = self.freeze_network(self.encoder_logv, freeze=True)

            # Unfreeze the decoder
            self.decoder_mu = self.freeze_network(self.decoder_mu, freeze=True)
 
    
    def score_matching(self, z, u, score_matching_coef=1, dag_coef=1., train=True):
        z.requires_grad_(True)
        
        logp = self.log_prob_z_c_u(z, u).sum()

        _dag_loss = self.dag_loss(z, u)
        
        grad1 = autograd.grad(logp, z, create_graph=True, retain_graph=True)[0]
        loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.

        loss2 = torch.zeros(z.shape[0], device=z.device)
        for i in range(z.shape[1]):
            if train:
                grad = autograd.grad(grad1[:, i].sum(), z, create_graph=True, retain_graph=True)[0][:, i]
            if not train:
                grad = autograd.grad(grad1[:, i].sum(), z, create_graph=False, retain_graph=True)[0][:, i]
                grad = grad.detach()
            loss2 += grad

        loss = score_matching_coef * (loss1 + loss2) - dag_coef * _dag_loss

        if not train:
            print("Score-matching loss is detached!")
            loss = loss.detach()

        return loss.sum()
    

    def elbo(self, x, u, len_dataset, a=1., b=1., c=1., d=1.):
        dec_mean, s_mean, s_var, z = self.forward(x, u)
        # import pdb;pdb.set_trace()
        batch_size, latent_dim = z.shape

        # Log( P(x|z) )
        logpx = log_normal(x, dec_mean, self.decoder_var.to(x.device)).sum(dim=-1)

        # Log( P(z|u) )
        logpz_cu =self.log_prob_z_c_u(z, u.detach()).sum()


        logqs_cux = log_normal(z, s_mean, s_var).sum(dim=-1)


        logqs_tmp = log_normal(z.view(batch_size, 1, latent_dim), s_mean.view(1, batch_size, latent_dim), s_var.view(1, batch_size, latent_dim))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(batch_size * len_dataset)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(batch_size * len_dataset)).sum(dim=-1)

        # print(f"prior is {logpz_cu}")
        
        elbo = -(a * logpx - b * (+logqs_cux - logpz_cu) ).mean() 
        
        return elbo, z