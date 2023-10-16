import time
import pickle
import pandas as pd

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data import CIVAEDataset
from metrics import mean_corr_coef as mcc
from models import cleanIVAE, cleanVAE, Discriminator, MLP, CIVAE, permute_dims

def get_model(data_dim, env_dim, latent_dim, config):
    if config.nf_ivae: 
        dim_input_mutual_effect = latent_dim
        output_dim_mutual_effect =  latent_dim * (latent_dim - 1)

    else :
        dim_input_mutual_effect = 4
        output_dim_mutual_effect =  1

    # Data Encoder
    data_encoder = MLP(
        input_dim = data_dim,
        output_dim = config.data_enc_output_dim,
        hidden_dims = config.data_enc_hidden_dim,
        activations = config.data_enc_activation,
    ).to(torch.float64).to(config.device)
    
    # Environment Encoder
    env_encoder = MLP(
        input_dim = env_dim,
        output_dim = config.env_enc_output_dim,
        hidden_dims = config.env_enc_hidden_dim,
        activations = config.env_enc_activation,
    ).to(torch.float64).to(config.device)
    
    # Linear Prior's Environment Encoder 
    env_encoder_linear_prior = MLP(
        input_dim = env_dim,
        output_dim = config.data_enc_output_dim,
        hidden_dims = config.env_encoder_linear_prior_hidden_dim,
        activations = config.env_encoder_linear_prior_activation,
    ).to(torch.float64).to(config.device)

    # Non-Linear Prior's Environment Encoder 
    env_encoder_non_linear_prior = MLP(
        input_dim = env_dim,
        output_dim = config.data_enc_output_dim,
        hidden_dims = config.env_encoder_non_linear_prior_hidden_dim,
        activations = config.env_encoder_non_linear_prior_activation,
    ).to(torch.float64).to(config.device)

    # Graph Weights
    graph_weights = MLP(
        input_dim = env_dim,
        output_dim = latent_dim * (latent_dim - 1),
        hidden_dims = config.graph_weights_hidden_dim,
        activations = config.graph_weights_activation,
    ).to(torch.float64).to(config.device)

    # Mutual Effects
    mutual_effects = MLP(
        input_dim = dim_input_mutual_effect ,
        output_dim = output_dim_mutual_effect,
        hidden_dims = config.mutual_effects_hidden_dim,
        activations = config.mutual_effects_activation,
    ).to(torch.float64).to(config.device)

    # CIVAE model
    civae = CIVAE(
        data_encoder=data_encoder,
        env_encoder=env_encoder,
        latent_dim = latent_dim,
        env_encoder_linear_prior = env_encoder_linear_prior,
        env_encoder_non_linear_prior = env_encoder_non_linear_prior,
        graph_weights_net = graph_weights,
        mutual_effects_net = mutual_effects,
        hidden_dims = config.civae_hidden_dim,
        activations = config.civae_activation,
        nf_ivae=config.nf_ivae
    ).to(torch.float64).to(config.device)

    # import pdb;pdb.set_trace()

    return civae


def runner(args, config):

    print("="*32)
    print(f"a: {config.a}")
    print(f"b: {config.b}")
    print(f"score_matching_coef: {config.score_matching_coef}")
    print(f"dag_coef: {config.dag_coef}")
    print("="*32)
    print(f"batch_size: {config.batch_size}")
    print(f"lr: {config.lr}")
    print("="*32)
    
    st = time.time()

    print('Executing script on: {}\n'.format(config.device))


    # Dataset and Dataloader
    dataset = CIVAEDataset(root=args.data_path, data_postfixe=config.data_postfix)
    d_data, d_aux, d_latent = dataset.get_dims()
    loader_params = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_params)

    perfs = []
    loss_hists = []
    perf_hists = []

    df_history = pd.DataFrame()

    for seed in range(args.seed, args.seed + args.n_sims):
        print(f"seed: {seed}")
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
        scheduler_elbo = optim.lr_scheduler.ReduceLROnPlateau(optimizer_elbo, factor=0.7, patience=2, verbose=True)

        optimizer_score_matching = optim.Adam(
            [
                *model.graph_weights.parameters(),
                *model.mutual_effects.parameters()
            ], 
            lr=config.lr
        )
        scheduler_score_matching = optim.lr_scheduler.ReduceLROnPlateau(optimizer_score_matching, factor=0.7, patience=2, verbose=True)


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
                model.optimization_freezer(phase=1)
                optimizer_elbo.zero_grad()
                elbo_loss, z = model.elbo(x, u, len(dataset), a=a, b=b, c=c, d=d)
                elbo_loss.backward(retain_graph=True)
                optimizer_elbo.step()


                # Score Matching Optimizaton
                model.optimization_freezer(phase=2)
                optimizer_score_matching.zero_grad()
                score_matching_loss = model.score_matching(z, u, score_matching_coef=config.score_matching_coef, dag_coef=config.dag_coef)
                score_matching_loss.backward(retain_graph=True)
                optimizer_score_matching.step()

                total_loss = elbo_loss.item() + score_matching_loss.item()
                train_loss += total_loss
                train_elbo_loss += elbo_loss.item()
                train_score_matching_loss += score_matching_loss.item()
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

            # with open(f"./ckpts/{epoch}.ckpt", 'wb') as handle:
            #     pickle.dump(model, handle)

            torch.save(model.state_dict(), f"./ckpts/{epoch}.ckpt")

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


            current_training = pd.DataFrame({
                "seed": [seed],
                "epoch": [epoch],
                "perf": [train_perf],
                "loss": [train_loss],
                "elbo": [train_elbo_loss],
                "score matching": [train_score_matching_loss],
                "elbo_lr": [optimizer_elbo.param_groups[0]['lr']],
                "sm_lr": [optimizer_score_matching.param_groups[0]['lr']]
            })

            df_history = pd.concat([df_history, current_training])

            df_history.to_csv("./reports.csv")


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
