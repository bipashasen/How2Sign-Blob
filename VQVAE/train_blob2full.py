
def main_blob2full(args):
    device = torch.device("cuda:{}".format(dist.get_local_rank()) if torch.cuda.is_available() else "cpu")

    args.distributed = dist.get_world_size() > 1

    training_data, validation_data, training_loader, validation_loader, x_train_var =\
        utils.load_data_and_data_loaders(args.dataset, args.batch_size, args.distributed)

    model = VQVAE_Blob2Full(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta, dist.get_local_rank()).to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt)['model'])

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    
    train_blob2full(args, training_loader, model, x_train_var, optimizer, device, dist)

def train_blob2full(args, training_loader, model, x_train_var, optimizer, device, dist):
    global save_idx_global, visual_folder, save_at, did

    visual_folder = visual_folder.format(args.dataset)

    os.makedirs(visual_folder, exist_ok=True)
    
    save_at = args.save_at
    did = args.device_id

    for i in range(args.n_updates):
        face, rhand, lhand, out = [x.to(device) for x in next(iter(training_loader))]
        optimizer.zero_grad()

        save_idx = None

        if i % save_at == 0 and dist.is_primary():
            save_idx = save_idx_global
            save_idx_global += 1

        embedding_loss, x_hat, perplexity = \
            model(x=(face, rhand, lhand, out), verbose=verbose, save_idx=save_idx, visual_folder=visual_folder)
        recon_loss = torch.mean((x_hat - out)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0 and dist.is_primary():
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, optimizer, results, hyperparameters, args.filename)

            print('Update #', i, 'Batches per epoch (',len(training_loader),')',
                  'Epoch (',int(i/len(training_loader)),')', 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))
