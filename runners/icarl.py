import time
import pickle
import pandas as pd

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data import ICARLDataset
from metrics import mean_corr_coef as mcc
from models import cleanIVAE, cleanVAE, Discriminator, MLP, MLPDoubleHead, ICARL, permute_dims

def get_model(data_dim, u_dim, latent_dim, config):

    print("#"*32)
    print(f"data_dim: {data_dim}")
    print(f"u_dim: {u_dim}")
    print(f"latent_dim: {latent_dim}")
    print("#"*32)

    # Encoder
    encoder = MLPDoubleHead(
        input_dim = data_dim + u_dim,
        head1_dim = latent_dim,
        head2_dim = latent_dim,
        hidden_dims = config.enc_hidden_dims,
        hidden_activations = config.enc_hidden_activations,
        head1_activation = config.enc_mu_activation,
        head2_activation = config.enc_logv_activation
    ).to(torch.float64).to(config.device)
    
    # λ_f 
    lambda_f = MLP(
        input_dim = u_dim,
        output_dim = 2 * latent_dim,
        hidden_dims = config.lambda_f_hidden_dims,
        activations = config.lambda_f_activations,
    ).to(torch.float64).to(config.device)

    # λ_NN
    lambda_nn = MLP(
        input_dim = u_dim,
        output_dim = int((latent_dim * (latent_dim-1))/2),
        hidden_dims = config.lambda_nn_hidden_dims,
        activations = config.lambda_nn_activations,
    ).to(torch.float64).to(config.device)

    # T_NN
    t_nn = MLP(
        input_dim = latent_dim,
        output_dim = int((latent_dim * (latent_dim -  1))/2),
        hidden_dims = config.t_nn_hidden_dims,
        activations = config.t_nn_activations
    ).to(torch.float64).to(config.device)

    # decoder
    decoder = MLP(
        input_dim = latent_dim,
        output_dim = data_dim,
        hidden_dims = config.decoder_hidden_dims,
        activations = config.decoder_activations
    ).to(torch.float64).to(config.device)

    # ICARL mdoel
    icarl = ICARL(
        data_dim = data_dim,
        u_dim = u_dim,
        latent_dim = latent_dim,
        lambda_f = lambda_f,
        lambda_nn = lambda_nn,
        t_nn = t_nn,
        encoder_net = encoder,
        decoder_net = decoder,
    ).to(torch.float64).to(config.device)

    return icarl


def runner(args, config):

    print("="*32)
    print(f"a: {config.a}")
    print(f"b: {config.b}")
    print(f"score_matching_coef: {config.score_matching_coef}")
    print(f"dag_coef: {config.dag_coef}")
    print("="*32)
    print(f"batch_size: {config.batch_size}")
    print(f"ELBO LR: {config.elbo_lr}")
    print(f"Score Matching LR: {config.score_matching_lr}")
    print("="*32)
    
    st = time.time()

    print('Executing script on: {}\n'.format(config.device))


    # Dataset and Dataloader
    dataset = ICARLDataset(root=args.data_path, data_postfix=config.data_postfix)
    data_dim, u_dim, latent_dim = dataset.get_dims()
    loader_params = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_params)

    perfs = []
    loss_hists = []
    perf_hists = []

    df_history = pd.DataFrame()

    for seed in range(args.seed, args.seed + args.n_sims):
        print(f"seed: {seed}")
        
        model = get_model(
            data_dim = data_dim, 
            u_dim = u_dim,  
            latent_dim = latent_dim, 
            config = config,
        )

        # ELBO Optimizer
        optimizer_elbo = optim.Adam(
            [
                *model.encoder_net.parameters(),
                *model.decoder_mu_net.parameters()
            ],
            lr = config.elbo_lr
        )
        scheduler_elbo = optim.lr_scheduler.ReduceLROnPlateau(optimizer_elbo, factor=0.7, patience=0, verbose=True)

        # Score Matching Optimizer
        optimizer_score_matching = optim.Adam(
            [
                *model.lambda_f.parameters(),
                *model.lambda_nn.parameters(),
                *model.t_nn.parameters() 
            ], 
            lr=config.score_matching_lr
        )
        scheduler_score_matching = optim.lr_scheduler.ReduceLROnPlateau(optimizer_score_matching, factor=0.7, patience=0, verbose=True)


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

            for i, (x, u, s_true) in enumerate(data_loader):
                
                x = x.to(config.device)
                u = u.to(config.device)
                s_true = s_true.to(config.device)
                
                # ELBO Optimization
                model.optimization_freezer(phase="elbo")
                optimizer_elbo.zero_grad()
                elbo_loss, enc_mu, enc_logv, z = model.elbo(x, u, len(dataset), a=a, b=b, c=c, d=d)
                elbo_loss.backward(retain_graph=True)
                optimizer_elbo.step()


                # Score Matching Optimizaton
                model.optimization_freezer(phase="score_matching")
                optimizer_score_matching.zero_grad()
                score_matching_loss = model.score_matching2(z, u, score_matching_coef=config.score_matching_coef, dag_coef=config.dag_coef)
                # print(f"score_matching_loss: {score_matching_loss}")
                score_matching_loss.backward(retain_graph=True)
                optimizer_score_matching.step()

                total_loss = elbo_loss.item() + score_matching_loss.item()
                train_loss += total_loss
                train_elbo_loss += elbo_loss.item()
                train_score_matching_loss += score_matching_loss.item()

                # print(f"ELBO: {elbo_loss.item()},\t Score Matching: {score_matching_loss.item()}")
                try:
                    perf = mcc(s_true.cpu().detach().numpy(), z.cpu().detach().numpy())
                except:
                    print("Error! Cannot compute performance!")
                    perf = 0
                train_perf += perf


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

    return perfs, loss_hists, perf_hists
