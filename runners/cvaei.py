import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data import SyntheticDataset
from data import CIVAEDataset
from metrics import mean_corr_coef as mcc
from models import cleanIVAE, cleanVAE, Discriminator, MLP, CIVAE, permute_dims

def get_model(data_dim, env_dim, latent_dim, config):

    # Data Encoder
    data_encoder = MLP(
        input_dim = data_dim,
        output_dim = config.data_enc_output_dim,
        hidden_dim = config.data_enc_hidden_dim,
        n_layers = config.data_enc_n_layers,
        activation = config.data_enc_activation,
        slope = config.data_enc_slope
    ).to(torch.float64).to(config.device)
    
    # Environment Encoder
    env_encoder = MLP(
        input_dim = env_dim,
        output_dim = config.env_enc_output_dim,
        hidden_dim = config.env_enc_hidden_dim,
        n_layers = config.env_enc_n_layers,
        activation = config.env_enc_activation,
        slope = config.env_enc_slope
    ).to(torch.float64).to(config.device)

    # Graph Weights
    graph_weights = MLP(
        input_dim = env_dim,
        output_dim = latent_dim * (latent_dim - 1) + 2 * latent_dim,
        hidden_dim = config.graph_weights_hidden_dim,
        n_layers = config.graph_weights_n_layers,
        activation = config.graph_weights_activation,
        slope = config.graph_weights_slope
    ).to(torch.float64).to(config.device)

    # Mutual Effects
    mutual_effects = MLP(
        input_dim = latent_dim,
        output_dim = latent_dim * (latent_dim - 1),
        hidden_dim = config.mutual_effects_hidden_dim,
        n_layers = config.mutual_effects_n_layers,
        activation = config.mutual_effects_activation,
        slope = config.mutual_effects_slope
    ).to(torch.float64).to(config.device)

    # CIVAE model
    civae = CIVAE(
        data_encoder=data_encoder,
        env_encoder=env_encoder,
        latent_dim = latent_dim,
        graph_weights_net = graph_weights,
        mutual_effects_net = mutual_effects,
        hidden_dim = config.civae_hidden_dim,
        n_layers = config.civae_n_layers,
        activation = config.civae_activation,
        slope = config.civae_slope
    ).to(torch.float64).to(config.device)

    return civae


def runner(args, config):
    st = time.time()

    print('Executing script on: {}\n'.format(config.device))

    #ivae dataset 
    factor = config.gamma > 0

    dset = SyntheticDataset(args.data_path, config.nps, config.ns, config.dl, config.dd, config.nl, config.s, config.p,
                            config.act, uncentered=config.uncentered, noisy=config.noisy, double=factor)
    d_data, d_latent, d_aux = dset.get_dims()

    loader_params = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = DataLoader(dset, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_params)

    # Dataset and Dataloader for CIVAE
    # dataset = CIVAEDataset(args.data_path)
    # d_data, d_aux, d_latent = dataset.get_dims()
    # loader_params = {'num_workers': 3, 'pin_memory': True} if torch.cuda.is_available() else {}
    # data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_params)

    perfs = []
    loss_hists = []
    perf_hists = []

    for seed in range(args.seed, args.seed + args.n_sims):
        # if config.ica:
        #     model = cleanIVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim,
        #                       n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
        # else:
        #     model = cleanVAE(data_dim=d_data, latent_dim=d_latent, hidden_dim=config.hidden_dim,
        #                      n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)

        model = get_model(
            data_dim = d_data,
            env_dim = d_aux,
            latent_dim = d_latent,
            config = config
        )

        elbo_params = [
            {"params": model.data_encoder.parameters()},
            {"params": model.env_encoder.parameters()},
        ]

        optimizer_elbo = optim.Adam(
            [
                *model.data_encoder.parameters(),
                *model.env_encoder.parameters(),
                *model.encoder_mu.parameters(),
                *model.encoder_logv.parameters(),
                *model.decoder_mu.parameters()
            ], 
            lr=config.lr
        )
        scheduler_elbo = optim.lr_scheduler.ReduceLROnPlateau(optimizer_elbo, factor=0.1, patience=0, verbose=True)

        optimizer_score_matching = optim.Adam(
            [
                *model.graph_weights.parameters(),
                *model.mutual_effects.parameters()
            ], 
            lr=config.lr
        )
        scheduler_score_matching = optim.lr_scheduler.ReduceLROnPlateau(optimizer_score_matching, factor=0.1, patience=0, verbose=True)


        # if factor:
        #     D = Discriminator(d_latent).to(config.device)
        #     optim_D = optim.Adam(D.parameters(), lr=config.lr,
        #                          betas=(.5, .9))

        loss_hist = []
        elbo_loss_hist = []
        score_matching_loss_hist = []
        perf_hist = []
        for epoch in range(1, config.epochs + 1):
            model.train()

            if config.anneal:
                a = config.a
                d = config.d
                b = config.b
                c = 0
                if epoch > config.epochs / 1.6:
                    b = 1
                    c = 1
                    d = 1
                    a = 2 * config.a
            else:
                a = config.a
                b = config.b
                c = config.c
                d = config.d

            train_loss = 0
            train_elbo_loss = 0
            train_score_matching_loss = 0
            train_perf = 0
            for i, data in enumerate(data_loader):
                # if not factor:
                #     x, u, s_true = data
                # else:
                # x, x2, u, s_true = data
                x, u, s_true = data
                x, u = x.to(config.device), u.to(config.device)
                
                # ELBO Optimization
                optimizer_elbo.zero_grad()
                elbo_loss, z = model.elbo(x, u, len(dataset), a=a, b=b, c=c, d=d)
                elbo_loss.backward(retain_graph=False)
                optimizer_elbo.step()


                # Score Matching Optimizaton
                optimizer_score_matching.zero_grad()
                score_matching_loss = model.score_matching(z.detach(), u.detach())
                # import pdb;pdb.set_trace()
                score_matching_loss.backward(retain_graph=True)
                optimizer_score_matching.step()

                total_loss = -elbo_loss.item() + score_matching_loss.item()
                train_loss += total_loss
                train_elbo_loss = -elbo_loss.item()
                train_score_matching_loss = score_matching_loss.item()
                try:
                    perf = mcc(s_true.numpy(), z.cpu().detach().numpy())
                except:
                    perf = 0
                train_perf += perf

                # optimizer.step()

                # if factor:
                #     ones = torch.ones(config.batch_size, dtype=torch.long, device=config.device)
                #     zeros = torch.zeros(config.batch_size, dtype=torch.long, device=config.device)
                #     x_true2 = x2.to(config.device)
                #     _, _, _, z_prime = model(x_true2)
                #     z_pperm = permute_dims(z_prime).detach()
                #     D_z_pperm = D(z_pperm)
                #     D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                #     optim_D.zero_grad()
                #     D_tc_loss.backward()
                #     optim_D.step()

            train_perf /= len(data_loader)
            perf_hist.append(train_perf)
            train_loss /= len(data_loader)
            loss_hist.append(train_loss)
            train_elbo_loss /= len(data_loader)
            elbo_loss_hist.append(train_elbo_loss)
            train_score_matching_loss /= len(data_loader)
            score_matching_loss_hist.append(train_score_matching_loss)
            print('==> Epoch {}/{}:\ttrain loss: {:.6f}\ttrain perf: {:.6f}\telbo: {:.6f}\tscore matching: {:.6f}'.format(epoch, config.epochs, train_loss,
                                                                                    train_perf, train_elbo_loss, train_score_matching_loss))

            if not config.no_scheduler:
                scheduler_elbo.step(train_elbo_loss)
                scheduler_score_matching.step(train_score_matching_loss)
        print('\ntotal runtime: {}'.format(time.time() - st))

        # evaluate perf on full dataset
        # Xt, Ut, St = dataset.x.to(config.device), dataset.u.to(config.device), dataset.s
        # if config.ica:
        #     _, _, _, s, _ = model(Xt, Ut)
        # else:
        #     _, _, _, s = model(Xt)
        # full_perf = mcc(dataset.s.numpy(), s.cpu().detach().numpy())
        # perfs.append(full_perf)
        # loss_hists.append(loss_hist)
        # perf_hists.append(perf_hist)

    return perfs, loss_hists, perf_hists
