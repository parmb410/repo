# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader

from diversify.auto_k_estimation import AutomatedKDataset

def prepare_subdomains(model, data_loaders, args):
    """
    Called before each training run (once per “round” or per full dataset pass).
    We’ll collect feature vectors per environment, run automated k‐estimation,
    and then use those new domain labels to feed into DIVERSIFY’s loss / alignment steps.
    """
    src_envs = []
    env_indices = []  # Keep track of how many samples per environment

    # 1) Loop over your existing source‐env dataloaders
    for env_loader in data_loaders:  
        all_feats = []
        for (x, y, idx) in env_loader:
            # 1.a) Extract features/embeddings for this batch (just like DIVERSIFY does originally)
            with torch.no_grad():
                feats = model.extract_features(x.cuda())  # shape: (batch_size, feat_dim)
            all_feats.append(feats.cpu().numpy())
        env_feats = np.concatenate(all_feats, axis=0)   # shape: (N_env, feat_dim)
        src_envs.append(env_feats)
        env_indices.append(env_feats.shape[0])

    # 2) Run automated K‐estimation across all concatenated features
    auto_k = AutomatedKDataset(src_envs, k_min=args.k_min, k_max=args.k_max)
    optimal_k, domain_labels_per_env = auto_k.run_estimation()
    print(f">>> Automatically chosen k = {optimal_k} based on silhouette score")

    # 3) domain_labels_per_env is now a list of NumPy arrays;
    #    each array has length = number of samples in that environment,
    #    and contains the assigned cluster label [0..(k−1)].
    #
    #    You can now feed these labels into DIVERSIFY’s “sub‐domain” logic,
    #    exactly in place of whatever manual k & label‐assignment you used before.
    #
    #    For example, you might assign a new `dlabels` tensor to your dataset object,
    #    or re‐build your “environment” dataloaders so that each sample is tagged
    #    with the cluster ID instead of the original environment index.

    return optimal_k, domain_labels_per_env
    
def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)
    if args.latent_domain_num < 6:
        args.batch_size = 32*args.latent_domain_num
    else:
        args.batch_size = 16*args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(
        args)

    best_valid_acc, target_acc = 0, 0

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    for round in range(args.max_epoch):
        print(f'\n========ROUND {round}========')
        print('====Feature update====')
        loss_list = ['class']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step]+[loss_result_dict[item]
                              for item in loss_list], colwidth=15)

        print('====Latent domain characterization====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step]+[loss_result_dict[item]
                              for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('====Domain-invariant feature learning====')

        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item+'_loss' for item in loss_list])
        print_key.extend([item+'_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)

        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                step_vals = algorithm.update(data, opt)

            results = {
                'epoch': step,
            }

            results['train_acc'] = modelopera.accuracy(
                algorithm, train_loader_noshuffle, None)

            acc = modelopera.accuracy(algorithm, valid_loader, None)
            results['valid_acc'] = acc

            acc = modelopera.accuracy(algorithm, target_loader, None)
            results['target_acc'] = acc

            for key in loss_list:
                results[key+'_loss'] = step_vals[key]
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            results['total_cost_time'] = time.time()-sss
            print_row([results[key] for key in print_key], colwidth=15)

    print(f'Target acc: {target_acc:.4f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
