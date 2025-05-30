import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class EMGDataset(Dataset):
    def __init__(self, root):
        self.data = np.load(os.path.join(root, 'emg_x.npy')).astype(np.float32)
        self.labels = np.load(os.path.join(root, 'emg_y.npy')).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y, idx  # <-- include index

def get_act_dataloader(args):
    dataset = EMGDataset(args.data_dir)

    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    valid_len = int(0.15 * total_len)
    test_len = total_len - train_len - valid_len

    train_set, valid_set, test_set = torch.utils.data.random_split(
        dataset, [train_len, valid_len, test_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    train_loader_noshuffle = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    target_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, train_loader_noshuffle, valid_loader, target_loader, None, None, None
