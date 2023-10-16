from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
import torch.autograd as autograd

from .my_mlp import MLP
from .utils import freeze_network
from .nets import log_normal

class Concater(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z):
        return torch.concat([z, z**2], dim=1)


class ICARL(nn.Module):
    def __init__(self,
                 data_dim,
                 latent_dim,
                 u_dim,
                 lambda_f,
                 lambda_nn,
                 t_nn,
                 encoder_net,
                 decoder_net):
        super().__init__()
        # Setting the dimensions
        self.data_dim = data_dim
        self.u_dim = u_dim
        self.latent_dim = latent_dim

        # Networks of ICARL
        # encoder is a mlp with two heads, one for "mu" and the second for "log(var"
        self.encoder_net = encoder_net

        # λ_f
        self.lambda_f = lambda_f

        # λ_NN
        self.lambda_nn = lambda_nn

        # T_NN
        self.t_nn = t_nn

        # Decodder
        self.decoder_mu_net = decoder_net
        self.decoder_logv = .01 * torch.ones(1)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        xu = torch.cat((x, u), dim=1)
        [mu, logv] = self.encoder_net(xu)
        return mu, logv.exp()

    def decoder(self, z):
        mu = self.decoder_mu_net(z)
        return mu, self.decoder_logv.exp()
    
    def logp_z_cey(self, z, u):
        z_z2 = torch.concat([z, z**2], dim=1)

        # I am not sure.
        out1 = torch.sum(
            self.lambda_f(u) * z_z2,
            dim = 1
        ).to(z.device)
        # out1 = torch.einsum(
        #     'ij,ij->i',
        #     self.lambda_f(ey),
        #     z_z2
        # )

        # I am not sure.
        out2 = torch.sum(
            self.lambda_nn(u) * self.t_nn(z),
            dim = 1
        ).to(z.device)
        # out2 = torch.einsum(
        #     'ij,ij->i',
        #     self.lambda_nn(ey),
        #     self.t_nn(z)
        # )

        return (out1 + out2)

    def optimization_freezer(self, phase):
        assert phase in ["elbo", "score_matching"], f"Error: Incorrect phase value for optimization_freezer: {phase}! The phase value of optimization_freezer must be one of this elements: ['elbo', 'score_matching']."
        if phase=="elbo":
            # UnFreeze the encoder and decoder and freeze other networks

            # λ_f
            self.lambda_f = freeze_network(self.lambda_f, requires_grad = False)

            # λ_NN
            self.lambda_nn = freeze_network(self.lambda_nn, requires_grad = False)

            # T_NN
            self.t_nn = freeze_network(self.t_nn, requires_grad = False)

            # encoder_net
            self.encoder_net = freeze_network(self.encoder_net, requires_grad = True)

            # Decodder
            self.decoder_mu_net = freeze_network(self.decoder_mu_net, requires_grad = True)
            
        else: 
            # Freeze the encoder and decoder and unfreeze other networks

            # λ_f
            self.lambda_f = freeze_network(self.lambda_f, requires_grad= True)

            # λ_NN
            self.lambda_nn = freeze_network(self.lambda_nn, requires_grad = True)

            # T_NN
            self.t_nn = freeze_network(self.t_nn, requires_grad = True)

            # encoder_net
            self.encoder_net = freeze_network(self.encoder_net, requires_grad = False)

            # Decodder
            self.decoder_mu_net = freeze_network(self.decoder_mu_net, requires_grad = False)
    
    def forward(self, x, u):
        [enc_mu, enc_logv] = self.encoder(x, u)
        z = self.reparameterize(enc_mu, enc_logv)
        out = self.decoder_mu_net(z)

        logp_z_cue = self.logp_z_cey(z, u)

        return out, enc_mu, enc_logv, z, logp_z_cue
    
    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        out, enc_mu, enc_logv, z, logp_z_cey = self.forward(x, u)
        M, d_latent = z.size()

        logpx = log_normal(x, out, self.decoder_logv.to(x.device)).sum(dim=-1)
        logqs_cxu = log_normal(z, enc_mu, enc_logv).sum(dim=-1)
        # logps_cu = log_normal(z, None, _p_z_cue).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), enc_mu.view(1, M, d_latent), enc_logv.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)


        elbo = -(a * logpx - b * (logqs_cxu - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logp_z_cey)).mean()
        # import pdb; pdb.set_trace()
        # elbo = -(a * logpx - b * (logqs_cxu - logp_z_cey)).mean()
        # print(f"logpx: {logpx}")
        # print(f"logqs_cxey: {logqs_cxey}")
        # print(f"logp_z_cey: {logp_z_cey}")
        # print("="*32)
        

        return elbo, enc_mu, enc_logv, z
    
    def score_matching(self, x, enc_mu, enc_logv, z, u, score_matching_coef=0.1, dag_coef=0, train=True):
        z.requires_grad_(True)
        u.requires_grad_(True)
        
        # logp = self.logp_z_cey(z, u)
        import pdb; pdb.set_trace()
        logqs_cxu = log_normal(z, enc_mu, enc_logv).sum(dim=-1)

        logp = logp.mean()
        grad_logp = autograd.grad(logp, z, create_graph=True, retain_graph=True)
        grad_logqs_cxu = autograd.grad(logqs_cxu.mean(), z, create_graph=True, retain_graph=True)

        loss = torch.norm(grad_logqs_cxu - grad_logp) ** 2
        return loss.mean()
    
    def score_matching3(self, z, u, score_matching_coef=1., dag_coef=0, train=True):
        z.requires_grad_(True)

        batch_size, z_dim = z.shape

        def mean_logp_z_cey(z, u):
            logp_z_cey = self.logp_z_cey(z, u)
            return torch.mean(logp_z_cey)
        
        jacobian = autograd.functional.jacobian(
            mean_logp_z_cey,
            (z, u),
            vectorize=True,
        )[0].requires_grad_(True)

        _hessian = autograd.functional.hessian(
            mean_logp_z_cey,
            (z, u),
            vectorize=True,
        )[0][0].requires_grad_(True)


        hessian = _hessian[
            torch.arange(batch_size),
            :,
            torch.arange(batch_size),
            :
        ]

        hessian_diag = hessian[
            :,
            torch.arange(z_dim),
            torch.arange(z_dim)
        ]


        loss1 = jacobian.pow(2) / 2
        loss2 =  hessian_diag



        loss = loss1 + loss2
        loss = torch.sum(loss, dim=1)

        return score_matching_coef * loss.mean()

        # import pdb; pdb.set_trace()

        # loss1 = torch.sum(jacobian ** 2 / 2) / batch_size
        # loss2 = torch.sum(_hessian) / batch_size

        return loss1 + loss2
        # loss2 = torch.trace(hessian)
        # loss2 = torch.sum(
        #     torch.vstack(
        #         [torch.trace(_hessian[i, :, i, :]) for i in range(batch_size)]
        #     )
        # )

        loss = loss1 + loss2
        loss = torch.sum(loss, dim=1)

        print(f"loss: {loss} {loss.mean()}")

        return loss.mean() 
    
    def score_matching2(self, z, u, score_matching_coef=1., dag_coef=0, train=True):
        z.requires_grad_(True)
        
        logp = self.logp_z_cey(z, u)
    
        # _dag_loss = self.dag_loss(z, u)
        
        # import pdb; pdb.set_trace()
        logp = logp.sum()
        grad1 = autograd.grad(logp, z, create_graph=True, retain_graph=True)[0]
        # loss1 = torch.norm(grad1, dim=-1) ** 2 / 2
        loss1 = torch.sum(grad1 ** 2 * 0.5, dim=-1)

        loss2 = torch.zeros(z.shape[0], device=z.device)
        for i in range(z.shape[1]):
            if train:
                grad = autograd.grad(grad1[:, i].sum(), z, create_graph=True, retain_graph=True)[0][:, i]
            if not train:
                grad = autograd.grad(grad1[:, i].sum(), z, create_graph=False, retain_graph=True)[0][:, i]
                grad = grad.detach()
            loss2 += grad

        loss =  score_matching_coef * (loss1 + loss2)    #- dag_coef * _dag_loss

        if not train:
            print("Score-matching loss is detached!")
            loss = loss.detach()

        return loss.mean()