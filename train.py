import numpy as np
import torch
import tqdm

import data_loader
import models


def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    model = model.to(device)

    loss_all = 0
    total = 0
    progress_format = 'train loss: {:6.6f}'
    err_counter = 0
    with tqdm.tqdm(total=len(loader), desc=progress_format.format(0)) as t:
        for i, data in enumerate(loader):
            # Concatenate all pairs to make all pair inference, this avoids embedding the surface twice
            names_0, names_1, pos_pairs_cas_arr, neg_pairs_cas_arr, geom_feats_0, geom_feats_1 = data
            if names_0 is None:
                continue
            all_pairs = torch.cat((pos_pairs_cas_arr, neg_pairs_cas_arr), dim=-3).to(device)
            labels = torch.cat((torch.ones(len(pos_pairs_cas_arr)), torch.zeros(len(neg_pairs_cas_arr)))).to(device)

            # Perform the learning
            try:
                optimizer.zero_grad()
                output = model(geom_feats_0, geom_feats_1, all_pairs)
                loss = criterion(output, labels)
                loss.backward()
                loss_all += loss.item() * len(labels)
                total += len(labels)
                optimizer.step()
            except Exception as e:
                print("--------------------")
                print(e)
                err_counter += 1
                print(f'Error counter: {err_counter} - {i // err_counter}')

            # stats
            t.set_description(progress_format.format(np.sqrt(loss_all / total)))
            t.update(1)

    return np.sqrt(loss_all / total)


if __name__ == '__main__':
    data_dir = './data/DIPS-split/data/train/'

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    dataset = data_loader.CNN3D_Dataset(data_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)
    model = models.SurfNet()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_loop(model=model, loader=data_loader, criterion=criterion, optimizer=optimizer, device=device)
