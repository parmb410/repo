# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader

from diversify.auto_k_estimation import AutomatedKDataset
import numpy as np
import torch

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)

    ##############################################
    # 1) BUILD / LOAD YOUR SOURCE-ENVIRONMENTS ETC #
    ##############################################

    # Example: load your source environment data loaders.
    # You should have a list or dict of DataLoaders for each source environment.
    # For illustration, let's assume:
    # src_env_loaders = [loader_env0, loader_env1, ..., loader_envN]
    # Replace this with your actual data-loading logic.
    src_env_loaders = []
    for env_id in args.source_envs:  # assume args.source_envs is a list of environment IDs
        loader = get_act_dataloader(env=env_id, batch_size=args.batch_size, shuffle=True)
        src_env_loaders.append(loader)

    # (Any other data loading or setup code goes here)
    ##############################################

    # –––––––––––––––––––––––––––––––––––––––––––––
    # 2) (NEW) RUN AUTOMATED K‐ESTIMATION ON EMBEDDINGS
    #    We assume that you already have “source environment” DataLoaders
    #    in a list called `src_env_loaders`. If your code loads them
    #    differently, adapt the names accordingly.

    src_envs = []
    for env_loader in src_env_loaders:
        # Collect all feature‐vectors (embeddings) for this environment
        all_feats = []
        for (x, y, idx) in env_loader:
            # You may need to replace `modelopera.extract_features`
            # with whatever your DIVERSIFY model uses to pull out a fixed‐dim embedding.
            with torch.no_grad():
                feats = modelopera.extract_features(x.cuda())
            all_feats.append(feats.cpu().numpy())
        if len(all_feats) > 0:
            env_feats = np.concatenate(all_feats, axis=0)
        else:
            env_feats = np.zeros((0, args.feature_dim))  # fallback if empty
        src_envs.append(env_feats)

    # Now run automated‐K on the concatenated features
    auto_k = AutomatedKDataset(src_envs, k_min=args.k_min, k_max=args.k_max)
    optimal_k, labels_per_env = auto_k.run_estimation()
    print(f">>> [AutomatedKDataset] Chosen k = {optimal_k} (silhouette‐based)")

    # labels_per_env is a list of numpy arrays, one per environment.
    # Each element labels_per_env[i] is an array of length = #samples in src_envs[i],
    # with integer cluster IDs in [0 .. optimal_k-1].

    # Concatenate all label arrays to a single 1D array of length = total samples
    all_labels_concat = np.concatenate(labels_per_env, axis=0)
    # Convert to a long PyTorch tensor and send to GPU if available
    all_labels_tensor = torch.from_numpy(all_labels_concat).long().cuda()

    # Now you need to overwrite however your code previously set `dlabels`.
    # For example, if your original DIVERSIFY code did something like:
    #     dataset.dlabels = torch.tensor(orig_environment_index).long().cuda()
    # Replace that with:
    #     dataset.dlabels = all_labels_tensor
    #
    # If your DataLoader iterates with (x, y, d), you must rebuild it so that
    # "d" comes from all_labels_tensor. Insert that assignment here.

    # ←── INSERT HERE: assign `all_labels_tensor` back into your per‐sample “domain labels” 
    #     wherever your training loop expects a `d` or `dlabels` input.

    #  End of Automated K block
    # –––––––––––––––––––––––––––––––––––––––––––––

    ##############################################
    # 3) PROCEED WITH THE REST OF DIVERSIFY TRAINING #
    ##############################################

    best_valid_acc = 0.0
    target_acc = 0.0

    # Example training loop (pseudocode). Replace with your actual loop.
    for round in range(args.n_rounds):
        # (Re)build any per-round structures, use all_labels_tensor as needed
        train_loader, valid_loader, test_loader = get_act_dataloader(env=round, batch_size=args.batch_size)

        model = modelopera.build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.n_epochs):
            model.train()
            for (x, y, d) in train_loader:
                x, y = x.cuda(), y.cuda()
                # Here, replace d with the automated domain label if needed
                # For example, d = all_labels_tensor[sample_index]
                outputs = model(x)
                loss = alg_loss_dict(outputs, y, d, args, model)  # pseudocode
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation step (pseudocode)
            model.eval()
            with torch.no_grad():
                # compute validation metrics
                valid_acc = 0.0  # replace with actual computation
                target_acc_local = 0.0

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                target_acc = target_acc_local

            # Print per-epoch stats
            results = {
                'valid_acc': valid_acc,
                'target_acc': target_acc_local,
                # ... other metrics ...
            }
            print_row([results[key] for key in ['valid_acc', 'target_acc']], colwidth=15)

    print(f'Target acc: {target_acc:.4f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
